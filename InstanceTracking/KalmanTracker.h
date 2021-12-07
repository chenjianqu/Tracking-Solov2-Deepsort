#ifndef KALMAN_H
#define KALMAN_H

#include <opencv2/video/tracking.hpp>

enum class TrackState {
    Tentative,
    Confirmed,
    Deleted
};


// This class represents the internel state of individual tracked objects observed as bounding box.
class KalmanTracker {
public:
    KalmanTracker();

    explicit KalmanTracker(cv::Rect2f initRect) : KalmanTracker() { init(initRect); }

    void init(cv::Rect2f initRect);

    void predict();

    void update(cv::Rect2f stateMat);

    void miss();

    [[nodiscard]] cv::Rect2f rect() const;

    [[nodiscard]] TrackState state() const { return _state; }

    [[nodiscard]] int id() const { return _id; }


    static constexpr int STATE_DIM = 7;
    static constexpr int MEASURE_DIM = 4;

private:

    inline static int count=0;

    TrackState _state = TrackState::Tentative;

    int _id = -1;

    int time_since_update = 0;
    int hits = 0;

    cv::KalmanFilter kf;
    cv::Mat measurement;
};

#endif //KALMAN_H