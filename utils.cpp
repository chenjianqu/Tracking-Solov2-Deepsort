//
// Created by chen on 2021/12/1.
//

#include "utils.h"


void draw_text(cv::Mat &img, const std::string &str, const cv::Scalar &color, const cv::Point& pos,  float scale, int thickness,bool reverse) {
    auto t_size = cv::getTextSize(str, cv::FONT_HERSHEY_SIMPLEX, scale, thickness, nullptr);
    cv::Point bottom_left, upper_right;
    if (reverse) {
        upper_right = pos;
        bottom_left = cv::Point(upper_right.x - t_size.width, upper_right.y + t_size.height);
    } else {
        bottom_left = pos;
        upper_right = cv::Point(bottom_left.x + t_size.width, bottom_left.y - t_size.height);
    }

    cv::rectangle(img, bottom_left, upper_right, color, -1);
    cv::putText(img, str, bottom_left, cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),thickness);
}

void draw_bbox(cv::Mat &img, const cv::Rect2f& bbox, const std::string &label, const cv::Scalar &color) {
    cv::rectangle(img, bbox, color);
    if (!label.empty()) {
        draw_text(img, label, color, bbox.tl());
    }
}



float getBoxIoU(const cv::Point2f &box1_minPt, const cv::Point2f &box1_maxPt,
                const cv::Point2f &box2_minPt, const cv::Point2f &box2_maxPt){

    cv::Point2f center1 = (box1_minPt+box1_maxPt)/2.f;
    cv::Point2f center2 = (box2_minPt+box2_maxPt)/2.f;
    float w1 = box1_maxPt.x - (float)box1_minPt.x;
    float h1 = box1_maxPt.y - (float)box1_minPt.y;
    float w2 = box2_maxPt.x - (float)box2_minPt.x;
    float h2 = box2_maxPt.y - (float)box2_minPt.y;

    if(std::abs(center1.x - center2.x) >= (w1/2+w2/2) && std::abs(center1.y - center2.y) >= (h1/2+h2/2)){
        return 0;
    }

    float inter_w = w1 + w2 - (std::max(center1.x + w1, center2.x + w2) - std::min(center1.x, center2.x));
    float inter_h = h1 + h2 - (std::max(center1.y + h1, center2.y + h2) - std::min(center1.y, center2.y));

    return (inter_h*inter_w) / (w1*h1 + w2*h2 - inter_h*inter_w);
}


/**
 * 计算两个box之间的IOU
 * @param bb_test
 * @param bb_gt
 * @return
 */
float getBoxIoU(const cv::Rect2f &bb_test, const cv::Rect2f &bb_gt) {
    auto in = (bb_test & bb_gt).area();
    auto un = bb_test.area() + bb_gt.area() - in;
    if (un <  DBL_EPSILON)
        return 0;
    return in / un;
}


cv::Scalar color_map(int64_t n) {
    auto bit_get = [](int64_t x, int64_t i) {
        return x & (1 << i);
    };

    int64_t r = 0, g = 0, b = 0;
    int64_t i = n;
    for (int64_t j = 7; j >= 0; --j) {
        r |= bit_get(i, 0) << j;
        g |= bit_get(i, 1) << j;
        b |= bit_get(i, 2) << j;
        i >>= 3;
    }
    return cv::Scalar(b, g, r);
}
