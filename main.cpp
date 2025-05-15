#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <filesystem>
#include "orb_detector.h"
#include "shape_detector.h"
#include "sign_detector.h"

std::vector<std::string> loadTemplatesFromDirectory(const std::string& directoryPath) {
    std::vector<std::string> templatePaths;

    try {
        if (!std::filesystem::exists(directoryPath)) {
            std::cerr << "Eroare: Directorul nu exista: " << directoryPath << std::endl;
            return templatePaths;
        }

        for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
            if (entry.is_regular_file()) {
                std::string extension = entry.path().extension().string();
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                    extension == ".bmp" || extension == ".tiff" || extension == ".webp") {
                    templatePaths.push_back(entry.path().string());
                    std::cout << "S-a incarcat template-ul: " << entry.path().string() << std::endl;
                }
            }
        }

        std::cout << "Total template-uri incarcate: " << templatePaths.size() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Eroare la incarcarea template-urilor: " << e.what() << std::endl;
    }

    return templatePaths;
}

int main() {
    std::string imagePath = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\pericol.jpeg)";
    cv::Mat inputImage = cv::imread(imagePath);

    if (inputImage.empty()) {
        std::cerr << "Eroare: Nu s-a putut incarca imaginea de la calea: " << imagePath << std::endl;
        return -1;
    }

    cv::namedWindow("Imagine originala", cv::WINDOW_NORMAL);
    cv::imshow("Imagine originala", inputImage);

    SignDetector signDetector;

    std::string templateDir = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\templates)";
    std::cout << "Incarcare template-uri din directorul: " << templateDir << std::endl;

    std::vector<std::string> templatePaths = loadTemplatesFromDirectory(templateDir);

    if (templatePaths.empty()) {
        std::cout << "Atentie: Nu s-au găsit template-uri! Metoda ORB nu va fi utilizată eficient." << std::endl;
        std::cout << "Poti crea un director 'templates' și adaugă semne de circulație pentru a îmbunătăți detecția." << std::endl;
    } else {
        signDetector.loadTemplates(templatePaths);
        std::cout << "Template-uri incarcate cu succes!" << std::endl;
    }

    std::vector<cv::Rect> detectedSigns = signDetector.detectSigns(inputImage);

    cv::Mat resultImage = inputImage.clone();
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