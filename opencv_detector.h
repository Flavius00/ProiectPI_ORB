#ifndef OPENCV_DETECTOR_H
#define OPENCV_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

enum ShapeType {
    CIRCLE,
    TRIANGLE,
    RECTANGLE,
    UNKNOWN
};

class ShapeDetector {
public:
    ShapeDetector();

    // Detectează contururi în imagine
    std::vector<std::vector<cv::Point>> detectContours(const cv::Mat& image);

    // Detectează formele din contururi
    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> detectShapes(
        const std::vector<std::vector<cv::Point>>& contours);

    // Funcții utilitare pentru detectarea formelor specifice
    ShapeType detectShape(const std::vector<cv::Point>& contour);
    bool isCircle(const std::vector<cv::Point>& contour, double& similarityScore);
    bool isTriangle(const std::vector<cv::Point>& contour, double& similarityScore);
    bool isRectangle(const std::vector<cv::Point>& contour, double& similarityScore);

private:
    // Parametri pentru detecția formelor
    double circularityThreshold;
    double triangularityThreshold;
    double rectangularityThreshold;

    // Funcții utilitare
    double calculateCircularity(const std::vector<cv::Point>& contour);
    double calculateTriangularity(const std::vector<cv::Point>& contour);
    double calculateRectangularity(const std::vector<cv::Point>& contour);
};

#endif