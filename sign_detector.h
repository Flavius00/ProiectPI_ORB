#ifndef SIGN_DETECTOR_H
#define SIGN_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "orb_detector.h"
#include "shape_detector.h"

class SignDetector {
public:
    SignDetector();

    // Detectează semne de circulație în imagine
    std::vector<cv::Rect> detectSigns(const cv::Mat& image);

    // Încarcă template-urile pentru semne de circulație
    void loadTemplates(const std::vector<std::string>& templatePaths);

private:
    // Detector de caracteristici ORB
    ORBDetector orbDetector;

    // Detector de forme
    ShapeDetector shapeDetector;

    // Template-uri pentru semne de circulație
    struct SignTemplate {
        cv::Mat image;
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        ShapeType shape;
    };

    std::vector<SignTemplate> templates;

    // Funcții utilitare pentru detecția semnelor
    std::vector<cv::Rect> detectSignsByColor(const cv::Mat& image);
    std::vector<cv::Rect> detectSignsByShape(const cv::Mat& image);
    std::vector<cv::Rect> detectSignsByFeatures(const cv::Mat& image);

    // Filtrează detecțiile suprapuse
    std::vector<cv::Rect> filterOverlappingDetections(const std::vector<cv::Rect>& detections, float overlapThreshold = 0.5);

    // Calculează IoU (Intersection over Union)
    float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);
};

#endif // SIGN_DETECTOR_H