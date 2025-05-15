#ifndef SHAPE_DETECTOR_H
#define SHAPE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum ShapeType {
    CIRCLE,
    TRIANGLE,
    RECTANGLE,
    UNKNOWN
};

class ShapeDetector {
public:
    ShapeDetector();

    std::vector<std::vector<cv::Point>> detectContours(const cv::Mat& image);
    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> detectShapes(
            const std::vector<std::vector<cv::Point>>& contours);

    ShapeType detectShape(const std::vector<cv::Point>& contour);
    bool isCircle(const std::vector<cv::Point>& contour, double& similarityScore);
    bool isTriangle(const std::vector<cv::Point>& contour, double& similarityScore);
    bool isRectangle(const std::vector<cv::Point>& contour, double& similarityScore);

private:
    double circularityThreshold;
    double triangularityThreshold;
    double rectangularityThreshold;

    double calculateCircularity(const std::vector<cv::Point>& contour);
    double calculateTriangularity(const std::vector<cv::Point>& contour);
    double calculateRectangularity(const std::vector<cv::Point>& contour);
};

#endif