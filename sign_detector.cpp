#include "sign_detector.h"
#include <iostream>

SignDetector::SignDetector() {
    // Initialize default ORB parameters if needed
    minMatchesRequired = 5;
    matchDistanceRatio = 0.8f;
}

std::vector<cv::Rect> SignDetector::detectSigns(const cv::Mat& image) {

    std::vector<cv::Rect> shapeDetections = detectSignsByShape(image);

    std::cout << "Detectii bazate pe forma: " << shapeDetections.size() << std::endl;

    std::vector<cv::Rect> allDetections;
    allDetections.insert(allDetections.end(), shapeDetections.begin(), shapeDetections.end());

    return filterOverlappingDetections(allDetections);
}

std::vector<cv::Rect> SignDetector::detectSignsByShape(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    std::vector<std::vector<cv::Point>> contours = shapeDetector.detectContours(image);

    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> shapes =
        shapeDetector.detectShapes(contours);

    for (const auto& shape : shapes) {
        ShapeType type = shape.first;
        const std::vector<cv::Point>& contour = shape.second;

        if (type == ShapeType::CIRCLE || type == ShapeType::TRIANGLE || type == ShapeType::RECTANGLE) {
            cv::Rect boundingRect = cv::boundingRect(contour);

            if (boundingRect.width >= 20 && boundingRect.height >= 20) {
                double area = cv::contourArea(contour);
                double perimeter = cv::arcLength(contour, true);

                if (perimeter > 0) {
                    double circularity = 4 * M_PI * area / (perimeter * perimeter);

                    if (circularity > 0.4) {
                        detections.push_back(boundingRect);
                    }
                }
            }
        }
    }

    return detections;
}

std::vector<cv::Rect> SignDetector::filterOverlappingDetections(
        const std::vector<cv::Rect>& detections, float overlapThreshold) {

    if (detections.empty()) {
        return std::vector<cv::Rect>();
    }

    std::vector<cv::Rect> sortedDetections = detections;

    std::sort(sortedDetections.begin(), sortedDetections.end(),
              [](const cv::Rect& a, const cv::Rect& b) {
                  return a.area() > b.area();
              });

    std::vector<bool> shouldKeep(sortedDetections.size(), true);

    for (size_t i = 0; i < sortedDetections.size(); i++) {
        if (!shouldKeep[i]) continue;

        const cv::Rect& rect1 = sortedDetections[i];

        for (size_t j = i + 1; j < sortedDetections.size(); j++) {
            if (!shouldKeep[j]) continue;

            const cv::Rect& rect2 = sortedDetections[j];

            float iou = calculateIoU(rect1, rect2);

            if (iou > overlapThreshold) {
                shouldKeep[j] = false;
            }
        }
    }

    std::vector<cv::Rect> filteredDetections;
    for (size_t i = 0; i < sortedDetections.size(); i++) {
        if (shouldKeep[i]) {
            filteredDetections.push_back(sortedDetections[i]);
        }
    }

    return filteredDetections;
}

// Helper function for calculating IoU (Intersection over Union)
float SignDetector::calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x1 >= x2 || y1 >= y2) {
        return 0.0f;
    }

    int intersectionArea = (x2 - x1) * (y2 - y1);
    int area1 = rect1.width * rect1.height;
    int area2 = rect2.width * rect2.height;
    int unionArea = area1 + area2 - intersectionArea;

    return static_cast<float>(intersectionArea) / unionArea;
}