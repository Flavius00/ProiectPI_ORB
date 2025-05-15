#include "shape_detector.h"

ShapeDetector::ShapeDetector() {
    circularityThreshold = 0.85;
    triangularityThreshold = 0.75;
    rectangularityThreshold = 0.8;
}

std::vector<std::vector<cv::Point>> ShapeDetector::detectContours(const cv::Mat& image) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

    cv::Mat edges;
    cv::Canny(blurredImage, edges, 50, 150);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> filteredContours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 100) {
            filteredContours.push_back(contour);
        }
    }

    return filteredContours;
}

std::vector<std::pair<ShapeType, std::vector<cv::Point>>> ShapeDetector::detectShapes(
        const std::vector<std::vector<cv::Point>>& contours) {

    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> shapes;

    for (const auto& contour : contours) {
        ShapeType shapeType = detectShape(contour);
        shapes.push_back(std::make_pair(shapeType, contour));
    }

    return shapes;
}

ShapeType ShapeDetector::detectShape(const std::vector<cv::Point>& contour) {
    double circularityScore = 0.0;
    double triangularityScore = 0.0;
    double rectangularityScore = 0.0;

    bool isCirc = isCircle(contour, circularityScore);
    bool isTri = isTriangle(contour, triangularityScore);
    bool isRect = isRectangle(contour, rectangularityScore);

    if (isCirc && circularityScore > triangularityScore && circularityScore > rectangularityScore) {
        return CIRCLE;
    }
    if (isTri && triangularityScore > circularityScore && triangularityScore > rectangularityScore) {
        return TRIANGLE;
    }
    if (isRect && rectangularityScore > circularityScore && rectangularityScore > triangularityScore) {
        return RECTANGLE;
    }

    return UNKNOWN;
}

bool ShapeDetector::isCircle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateCircularity(contour);
    return similarityScore > circularityThreshold;
}

bool ShapeDetector::isTriangle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateTriangularity(contour);
    return similarityScore > triangularityThreshold;
}

bool ShapeDetector::isRectangle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateRectangularity(contour);
    return similarityScore > rectangularityThreshold;
}

double ShapeDetector::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);

    double circularity = (4 * M_PI * area) / (perimeter * perimeter);

    return circularity;
}

double ShapeDetector::calculateTriangularity(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point> approx;
    double epsilon = 0.04 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    if (approx.size() != 3) {
        return 0.0;
    }

    double contourArea = cv::contourArea(contour);
    double triangleArea = cv::contourArea(approx);

    if (triangleArea == 0) {
        return 0.0;
    }

    return contourArea / triangleArea;
}

double ShapeDetector::calculateRectangularity(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point> approx;
    double epsilon = 0.04 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    if (approx.size() != 4) {
        return 0.0;
    }

    cv::RotatedRect boundingRect = cv::minAreaRect(contour);
    double boundingRectArea = boundingRect.size.width * boundingRect.size.height;

    double contourArea = cv::contourArea(contour);

    if (boundingRectArea == 0) {
        return 0.0;
    }

    return contourArea / boundingRectArea;
}