#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>
#include "shape_detector.h"
#include "sign_detector.h"
#include "opencv_detector.h"
#include <iomanip>

float calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x_left = std::max(rect1.x, rect2.x);
    int y_top = std::max(rect1.y, rect2.y);
    int x_right = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y_bottom = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    if (x_right <= x_left || y_bottom <= y_top)
        return 0.0f;

    int intersectionArea = (x_right - x_left) * (y_bottom - y_top);
    int unionArea = rect1.area() + rect2.area() - intersectionArea;

    std::cout << "Area difference: " << (float)abs(rect1.area() - rect2.area()) / std::max(rect1.area(), rect2.area()) << std::endl;

    return static_cast<float>(intersectionArea) / unionArea;
}

void evaluateDetections(
    const std::vector<cv::Rect>& groundTruth,
    const std::vector<cv::Rect>& predictions,
    const std::vector<float>& thresholds = {0.5f, 0.75f}
) {
    std::cout << std::fixed << std::setprecision(4);

    for (float iouThreshold : thresholds) {
        int TP = 0, FP = 0, FN = 0;
        float totalIoU = 0.0;
        int matched = 0;
        std::vector<bool> matchedPredictions(predictions.size(), false);

        for (const auto& gt : groundTruth) {
            bool foundMatch = false;
            for (size_t i = 0; i < predictions.size(); ++i) {
                if (matchedPredictions[i]) continue;
                float iou = calculateIoU(gt, predictions[i]);
                if (iou >= iouThreshold) {
                    TP++;
                    totalIoU += iou;
                    matchedPredictions[i] = true;
                    matched++;
                    foundMatch = true;
                    break;
                }
            }
            if (!foundMatch) FN++;
        }

        for (size_t i = 0; i < predictions.size(); ++i) {
            if (!matchedPredictions[i]) FP++;
        }

        float precision = TP + FP > 0 ? TP / static_cast<float>(TP + FP) : 0;
        float recall = TP + FN > 0 ? TP / static_cast<float>(TP + FN) : 0;
        float f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        float mIoU = matched > 0 ? totalIoU / matched : 0;
        float missRate = FN + TP > 0 ? FN / static_cast<float>(FN + TP) : 0;

        std::cout << "\n--- IoU Threshold = " << iouThreshold << " ---" << std::endl;
        std::cout << "Precision:  " << precision << std::endl;
        std::cout << "Recall:     " << recall << std::endl;
        std::cout << "F1-Score:   " << f1 << std::endl;
        std::cout << "mean IoU:   " << mIoU << std::endl;
        std::cout << "AP@" << iouThreshold << ":  " << precision << std::endl;
        std::cout << "Miss Rate:  " << missRate << std::endl;
    }
}


int main() {
    std::string imagePath = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\pericol.jpeg)";
    cv::Mat inputImage = cv::imread(imagePath);

    if (inputImage.empty()) {
        std::cerr << "Eroare: Nu s-a putut incarca imaginea de la calea: " << imagePath << std::endl;
        return -1;
    }

    // ======= CUSTOM DETECTION VIA SignDetector =======
    SignDetector customSignDetector;
    std::vector<cv::Rect> customBoxes = customSignDetector.detectSigns(inputImage);


    // ======= OPENCV DETECTION (GROUND TRUTH) =======
    OpenCVShapeDetector opencvDetector;
    std::vector<std::vector<cv::Point>> opencvContours = opencvDetector.detectContours(inputImage);
    auto opencvShapes = opencvDetector.detectShapes(opencvContours);
    std::vector<cv::Rect> opencvBoxes;
    for (const auto& shape : opencvShapes) {
        if (shape.first != ShapeType::UNKNOWN) {
            opencvBoxes.push_back(cv::boundingRect(shape.second));
        }
    }

    // ======= DRAW RESULTS =======
    cv::Mat customImage = inputImage.clone();
    for (const auto& rect : customBoxes) {
        cv::rectangle(customImage, rect, cv::Scalar(0, 255, 0), 2);
        cv::putText(customImage, "Custom", rect.tl() - cv::Point(0, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

    cv::Mat opencvImage = inputImage.clone();
    for (const auto& rect : opencvBoxes) {
        cv::rectangle(opencvImage, rect, cv::Scalar(255, 0, 0), 2);
        cv::putText(opencvImage, "OpenCV", rect.tl() - cv::Point(0, 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }

    // ======= SHOW IMAGES =======
    cv::namedWindow("Custom Detection", cv::WINDOW_NORMAL);
    cv::imshow("Custom Detection", customImage);

    cv::namedWindow("OpenCV Detection", cv::WINDOW_NORMAL);
    cv::imshow("OpenCV Detection", opencvImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    std::cout << "Detected by Custom: " << customBoxes.size() << std::endl;
    std::cout << "Detected by OpenCV: " << opencvBoxes.size() << std::endl;

    evaluateDetections(opencvBoxes, customBoxes);

    return 0;
}