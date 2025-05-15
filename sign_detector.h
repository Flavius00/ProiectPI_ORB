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

    // Setează parametrii pentru detecția ORB
    void setORBParameters(int minMatches = 5, float matchRatio = 0.8f) {
        minMatchesRequired = minMatches;
        matchDistanceRatio = matchRatio;
    }

private:
    // Detector de caracteristici ORB
    ORBDetector orbDetector;

    // Detector de forme
    ShapeDetector shapeDetector;

    // Parametri pentru detecția bazată pe ORB
    int minMatchesRequired;
    float matchDistanceRatio;

    // Template-uri pentru semne de circulație
    struct SignTemplate {
        cv::Mat image;
        std::vector<KeyPoint> keypoints;
        cv::Mat descriptors;
        ShapeType shape;
        std::string originalPath; // Pentru debugging
    };

    std::vector<SignTemplate> templates;

    // Funcții utilitare pentru detecția semnelor
    std::vector<cv::Rect> detectSignsByColor(const cv::Mat& image);
    std::vector<cv::Rect> detectSignsByShape(const cv::Mat& image);

    // Funcții de preprocesare
    void preprocessTemplate(const cv::Mat& input, cv::Mat& output);
    void preprocessImage(const cv::Mat& input, cv::Mat& output);

    // Filtrează detecțiile suprapuse
    std::vector<cv::Rect> filterOverlappingDetections(const std::vector<cv::Rect>& detections, float overlapThreshold = 0.5);

    // Calculează IoU (Intersection over Union)
    float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2);

protected:
    std::vector<cv::Rect> detectSignsByFeatures(const cv::Mat& image);
};

#endif // SIGN_DETECTOR_H