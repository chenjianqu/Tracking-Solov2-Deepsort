#include <algorithm>

#include "DeepSORT.h"
#include "Extractor.h"
#include "TrackerManager.h"

using namespace std;
using namespace cv;


namespace {
    float iou(const Rect2f &bb_test, const Rect2f &bb_gt) {
        auto in = (bb_test & bb_gt).area();
        auto un = bb_test.area() + bb_gt.area() - in;

        if (un < DBL_EPSILON)
            return 0;

        return in / un;
    }
}

torch::Tensor iou_dist(const vector<Rect2f> &dets, const vector<Rect2f> &trks) {
    auto trk_num = trks.size();
    auto det_num = dets.size();
    auto dist = torch::empty({int64_t(trk_num), int64_t(det_num)});
    for (size_t i = 0; i < trk_num; i++) // compute iou matrix as a distance matrix
        {
        for (size_t j = 0; j < det_num; j++) {
            dist[i][j] = 1 - iou(trks[i], dets[j]);
        }
        }
    return dist;
}





DeepSORT::DeepSORT(const array<int64_t, 2> &dim)
        : extractor(make_unique<Extractor>()),
          manager(make_unique<TrackerManager<TrackData>>(data, dim)),
          feat_metric(make_unique<FeatureMetric<TrackData>>(data)) {

}



vector<InstInfo> DeepSORT::update(const std::vector<InstInfo> &detections, cv::Mat ori_img) {
    manager->predict();
    manager->remove_nan();

    auto matched = manager->update(
            detections,
            [this, &detections, &ori_img](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(data[t].kalman.rect());
                }
                vector<cv::Mat> boxes;
                vector<cv::Rect2f> dets;
                for (auto d:det_ids) {
                    dets.push_back(detections[d].rect);
                    boxes.push_back(ori_img(detections[d].rect));
                }

                auto iou_mat = iou_dist(dets, trks);
                auto feat_mat = feat_metric->distance(extractor->extract(boxes), trk_ids);
                feat_mat.masked_fill_((iou_mat > 0.8f).__ior__(feat_mat > 0.2f), INVALID_DIST);
                return feat_mat;
            },
            [this, &detections](const std::vector<int> &trk_ids, const std::vector<int> &det_ids) {
                vector<cv::Rect2f> trks;
                for (auto t : trk_ids) {
                    trks.push_back(data[t].kalman.rect());
                }
                vector<cv::Rect2f> dets;
                for (auto &d:det_ids) {
                    dets.push_back(detections[d].rect);
                }
                auto iou_mat = iou_dist(dets, trks);
                iou_mat.masked_fill_(iou_mat > 0.7f, INVALID_DIST);
                return iou_mat;
            });

    vector<cv::Mat> boxes;
    vector<int> targets;
    for (auto[x, y]:matched) {
        targets.emplace_back(x);
        boxes.emplace_back(ori_img(detections[y].rect));
    }
    feat_metric->update(extractor->extract(boxes), targets);

    manager->remove_deleted();

    //return manager->visible_tracks();
    return manager->visible_tracks_info();
}
