#ifndef SIGN_DETECTOR_H
#define SIGN_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "orb_detector.h"
#include "shape_detector.h"

class SignDetector {
public:
    SignDetector();

    std::vector<cv::Rect> detectSigns(const cv::Mat& image);

    void loadTemplates(const std::vector<std::string>& templatePaths);

    void setORBParameters(int minMatches = 5, float matchRatio = 0.8f) {
        minMatchesRequired = minMatches;
        matchDistanceRatio = matchRatio;
    }

private:
    ORBDetector orbDetector;

    ShapeDetector shapeDetector;

    int minMatchesRequired;
    float matchDistanceRatio;

    struct SignTemplate {
        cv::Mat image;
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        ShapeType shape;
        std::string originalPath;
    };

    std::vector<SignTemplate> templates;

    std::vector<cv::Rect> detectSignsByShape(const cv::Mat& image);
    std::vector<cv::Rect> detectSignsByFeatures(const cv::Mat& image);

    void preprocessTemplate(const cv::Mat& input, cv::Mat& output);
    void preprocessImage(const cv::Mat& input, cv::Mat& output);

    std::vector<cv::Rect> filterOverlappingDetections(const std::vector<cv::Rect>& detections, float overlapThreshold = 0.5);

    float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);
};

#endif