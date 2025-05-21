#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>
#include "shape_detector.h"
#include "sign_detector.h"


float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x1 >= x2 || y1 >= y2) return 0.0f;

    int intersectionArea = (x2 - x1) * (y2 - y1);
    int unionArea = rect1.area() + rect2.area() - intersectionArea;

    return static_cast<float>(intersectionArea) / unionArea;
}

void computeMetrics(const std::vector<cv::Rect>& opencvResults, const std::vector<cv::Rect>& customResults, float iouThreshold = 0.5f) {
    int TP = 0, FP = 0, FN = 0;
    std::vector<bool> matchedCustom(customResults.size(), false);

    for (const auto& refBox : opencvResults) {
        bool matched = false;
        for (size_t i = 0; i < customResults.size(); ++i) {
            if (!matchedCustom[i] && calculateIoU(refBox, customResults[i]) > iouThreshold) {
                TP++;
                matchedCustom[i] = true;
                matched = true;
                break;
            }
        }
        if (!matched) FN++;
    }

    for (size_t i = 0; i < customResults.size(); ++i) {
        if (!matchedCustom[i]) FP++;
    }

    float precision = TP + FP > 0 ? TP / static_cast<float>(TP + FP) : 0;
    float recall = TP + FN > 0 ? TP / static_cast<float>(TP + FN) : 0;
    float f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "F1-Score: " << f1 << std::endl;
}


int main() {
    std::string imagePath = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\cedeaza.jpeg)";
    cv::Mat inputImage = cv::imread(imagePath);

    if (inputImage.empty()) {
        std::cerr << "Eroare: Nu s-a putut incarca imaginea de la calea: " << imagePath << std::endl;
        return -1;
    }

    cv::namedWindow("Imagine originala", cv::WINDOW_NORMAL);
    cv::imshow("Imagine originala", inputImage);

    SignDetector signDetector;

    std::vector<cv::Rect> detectedSigns = signDetector.detectSigns(inputImage);

    cv::Mat resultImage = inputImage.clone();

    SignDetector customDetector;
    ShapeDetector opencvShapeDetector;

    std::vector<cv::Rect> customDetections = customDetector.detectSigns(inputImage);

    std::vector<std::vector<cv::Point>> contours = opencvShapeDetector.detectContours(inputImage);
    auto shapes = opencvShapeDetector.detectShapes(contours);

    std::vector<cv::Rect> opencvDetections;
    for (const auto& shape : shapes) {
        if (shape.first != UNKNOWN) {
            cv::Rect box = cv::boundingRect(shape.second);
            opencvDetections.push_back(box);
        }
    }

    computeMetrics(opencvDetections, customDetections);

    for (const auto& signRect : detectedSigns) {
        cv::rectangle(resultImage, signRect, cv::Scalar(0, 255, 0), 2);

        std::string label = "Semn detectat";
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        cv::putText(resultImage, label, cv::Point(signRect.x, signRect.y - 5),
                    fontFace, fontScale, cv::Scalar(0, 255, 0), thickness);
    }

    cv::namedWindow("Semne detectate", cv::WINDOW_NORMAL);
    cv::imshow("Semne detectate", resultImage);

    std::cout << "S-au detectat " << detectedSigns.size() << " semne de circulatie." << std::endl;

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}