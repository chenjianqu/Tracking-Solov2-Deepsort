#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>

#include "Track.h"
#include "Extractor.h"
#include "TrackerManager.h"
#include "../utils.h"


torch::Tensor iou_dist(const std::vector<cv::Rect2f> &dets, const std::vector<cv::Rect2f> &trks);

// save features of the track in GPU
class FeatureBundle {
public:
    FeatureBundle() : full(false), next(0), store(torch::empty({budget, feat_dim}).cuda()) {}

    void clear() {
        next = 0;
        full = false;
    }

    [[nodiscard]] bool empty() const {
        return next == 0 && !full;
    }

    void add(torch::Tensor feat) {
        if (next == budget) {
            full = true;
            next = 0;
        }
        store[next++] = feat;
    }

    [[nodiscard]] torch::Tensor get() const {
        return full ? store : store.slice(0, 0, next);
    }

private:
    static const int64_t budget = 100, feat_dim = 512;

    bool full;
    int64_t next;

    torch::Tensor store;
};



template<typename TrackData>
class FeatureMetric {
public:
    explicit FeatureMetric(std::vector<TrackData> &data) : data(data) {}

    torch::Tensor distance(torch::Tensor features, const std::vector<int> &targets) {
        auto dist = torch::empty({int64_t(targets.size()), features.size(0)});
        if (features.size(0)) {
            for (size_t i = 0; i < targets.size(); ++i) {
                dist[i] = nn_cosine_distance(data[targets[i]].feats.get(), features);
            }
        }

        return dist;
    }

    void update(torch::Tensor feats, const std::vector<int> &targets) {
        for (size_t i = 0; i < targets.size(); ++i) {
            data[targets[i]].feats.add(feats[i]);
        }
    }

private:
    std::vector<TrackData> &data;

    torch::Tensor nn_cosine_distance(torch::Tensor x, torch::Tensor y) {
        return std::get<0>(torch::min(1 - torch::matmul(x, y.t()), 0)).cpu();
    }
};




class DeepSORT {
public:
    using Ptr=std::unique_ptr<DeepSORT>;
    explicit DeepSORT(const std::array<int64_t, 2> &dim);

    std::vector<InstInfo> update(const std::vector<InstInfo> &detections, cv::Mat ori_img);

private:
    struct TrackData {
        KalmanTracker kalman;
        FeatureBundle feats;
        InstInfo info;
    };

    std::vector<TrackData> data;
    std::unique_ptr<Extractor> extractor;
    std::unique_ptr<TrackerManager<TrackData>> manager;
    std::unique_ptr<FeatureMetric<TrackData>> feat_metric;
};


#endif //DEEPSORT_H
