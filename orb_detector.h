#ifndef ORB_DETECTOR_H
#define ORB_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class KeyPoint {
public:
    KeyPoint() : x(0), y(0), response(0.0f), angle(0.0f), size(0.0f) {}
    KeyPoint(int _x, int _y, float _response = 0.0f, float _angle = 0.0f, float _size = 0.0f)
            : x(_x), y(_y), response(_response), angle(_angle), size(_size) {}

    int x, y;
    float response;
    float angle;
    float size;
};

class ORBDetector {
public:
    ORBDetector(int nfeatures = 500, float scaleFactor = 1.2f, int nlevels = 8);

    std::vector<KeyPoint> detect(const cv::Mat& image);

    cv::Mat compute(const cv::Mat& image, std::vector<KeyPoint>& keypoints);

    std::vector<KeyPoint> detectAndCompute(const cv::Mat& image, cv::Mat& descriptors);

    std::vector<std::pair<int, int>> match(const cv::Mat& descriptors1,
                                           const cv::Mat& descriptors2,
                                           float threshold = 0.75f);

private:
    int nfeatures;
    float scaleFactor;
    int nlevels;

    std::vector<KeyPoint> detectFAST(const cv::Mat& image, int threshold = 20);

    void computeOrientation(const cv::Mat& image, std::vector<KeyPoint>& keypoints);

    float calculateCenteredMoment(const cv::Mat& image, int p, int q, int cx, int cy, int radius);

    cv::Mat computeBRIEF(const cv::Mat& image, std::vector<KeyPoint>& keypoints);

    std::vector<std::pair<cv::Point, cv::Point>> generateBriefPattern(int patchSize = 31, int numPairs = 256);

    int hammingDistance(const cv::Mat& desc1, const cv::Mat& desc2);
};

#endif