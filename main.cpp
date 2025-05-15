#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include "orb_detector.h"
#include "shape_detector.h"
#include "sign_detector.h"

int main() {
    // Încărcarea imaginii hardcodate
    std::string imagePath = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\stop.jpeg)";
    cv::Mat inputImage = cv::imread(imagePath);

    if (inputImage.empty()) {
        std::cerr << "Eroare: Nu s-a putut încărca imaginea de la calea: " << imagePath << std::endl;
        return -1;
    }

    // Afișarea imaginii originale
    cv::namedWindow("Imagine originala", cv::WINDOW_NORMAL);
    cv::imshow("Imagine originala", inputImage);

    // Crearea detectorului de semne
    SignDetector signDetector;

    // Detectarea semnelor
    std::vector<cv::Rect> detectedSigns = signDetector.detectSigns(inputImage);

    // Marcarea semnelor detectate pe imaginea originală
    cv::Mat resultImage = inputImage.clone();
    for (const auto& signRect : detectedSigns) {
        // Desenarea unui dreptunghi în jurul semnului detectat
        cv::rectangle(resultImage, signRect, cv::Scalar(0, 255, 0), 2);
    }

    // Afișarea imaginii rezultat
    cv::namedWindow("Semne detectate", cv::WINDOW_NORMAL);
    cv::imshow("Semne detectate", resultImage);

    // Salvarea imaginii rezultat
    cv::imwrite("detected_signs.jpg", resultImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}