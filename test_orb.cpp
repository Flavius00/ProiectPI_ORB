#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "orb_detector.h"
#include "sign_detector.h"

// Funcție pentru a afișa potrivirile ORB între template și imagine
void displayORBMatches(const cv::Mat& image, const std::string& templatePath) {
    // Încărcăm template-ul
    cv::Mat templateImage = cv::imread(templatePath);
    if (templateImage.empty()) {
        std::cerr << "Nu s-a putut încărca template-ul: " << templatePath << std::endl;
        return;
    }

    // Creăm detectorul ORB
    ORBDetector orbDetector(1000, 1.2f, 8); // Parametri îmbunătățiți

    // Detectăm caracteristicile în imagine
    cv::Mat imageGray;
    if (image.channels() == 3) {
        cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);
    } else {
        imageGray = image.clone();
    }
    cv::equalizeHist(imageGray, imageGray);

    cv::Mat imageDescriptors;
    std::vector<KeyPoint> imageKeypoints = orbDetector.detectAndCompute(imageGray, imageDescriptors);

    // Detectăm caracteristicile în template
    cv::Mat templateGray;
    if (templateImage.channels() == 3) {
        cv::cvtColor(templateImage, templateGray, cv::COLOR_BGR2GRAY);
    } else {
        templateGray = templateImage.clone();
    }
    cv::equalizeHist(templateGray, templateGray);

    cv::Mat templateDescriptors;
    std::vector<KeyPoint> templateKeypoints = orbDetector.detectAndCompute(templateGray, templateDescriptors);

    // Potrivim descriptorii
    std::vector<std::pair<int, int>> matches = orbDetector.match(imageDescriptors, templateDescriptors, 0.8f);

    std::cout << "S-au găsit " << matches.size() << " potriviri între imagine și template." << std::endl;

    // Afișăm imaginea cu punctele cheie
    cv::Mat imgWithKeypoints;
    cv::cvtColor(imageGray, imgWithKeypoints, cv::COLOR_GRAY2BGR);

    for (const auto& kp : imageKeypoints) {
        cv::circle(imgWithKeypoints, cv::Point(kp.x, kp.y), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::namedWindow("Puncte cheie imagine", cv::WINDOW_NORMAL);
    cv::imshow("Puncte cheie imagine", imgWithKeypoints);

    // Afișăm template-ul cu punctele cheie
    cv::Mat templateWithKeypoints;
    cv::cvtColor(templateGray, templateWithKeypoints, cv::COLOR_GRAY2BGR);

    for (const auto& kp : templateKeypoints) {
        cv::circle(templateWithKeypoints, cv::Point(kp.x, kp.y), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::namedWindow("Puncte cheie template", cv::WINDOW_NORMAL);
    cv::imshow("Puncte cheie template", templateWithKeypoints);

    // Vizualizăm potrivirile
    cv::Mat matchesImage;

    // Redimensionăm imaginea și template-ul pentru afișare
    double scale = 600.0 / std::max(image.rows, image.cols);
    cv::Mat resizedImage, resizedTemplate;
    cv::resize(image, resizedImage, cv::Size(), scale, scale);

    double templateScale = 300.0 / std::max(templateImage.rows, templateImage.cols);
    cv::resize(templateImage, resizedTemplate, cv::Size(), templateScale, templateScale);

    // Construim imaginea rezultat
    int resultHeight = std::max(resizedImage.rows, resizedTemplate.rows);
    int resultWidth = resizedImage.cols + resizedTemplate.cols;

    matchesImage = cv::Mat::zeros(resultHeight, resultWidth, CV_8UC3);

    // Copiem imaginea și template-ul
    resizedImage.copyTo(matchesImage(cv::Rect(0, 0, resizedImage.cols, resizedImage.rows)));
    resizedTemplate.copyTo(matchesImage(cv::Rect(resizedImage.cols, 0, resizedTemplate.cols, resizedTemplate.rows)));

    // Desenăm liniile de potrivire
    for (const auto& match : matches) {
        cv::Point pt1(static_cast<int>(imageKeypoints[match.first].x * scale),
                      static_cast<int>(imageKeypoints[match.first].y * scale));

        cv::Point pt2(static_cast<int>(templateKeypoints[match.second].x * templateScale) + resizedImage.cols,
                      static_cast<int>(templateKeypoints[match.second].y * templateScale));

        cv::line(matchesImage, pt1, pt2, cv::Scalar(0, 255, 255), 1);
    }

    cv::namedWindow("Potriviri ORB", cv::WINDOW_NORMAL);
    cv::imshow("Potriviri ORB", matchesImage);

    cv::waitKey(0);
}

// Funcție pentru a testa detecția exclusiv cu ORB
void testORBDetection(const std::string& imagePath, const std::vector<std::string>& templatePaths) {
    // Încărcăm imaginea de test
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Nu s-a putut încărca imaginea: " << imagePath << std::endl;
        return;
    }

    // Afișăm imaginea originală
    cv::namedWindow("Imagine originală", cv::WINDOW_NORMAL);
    cv::imshow("Imagine originală", image);

    // Creăm un detector de semne special pentru ORB
    SignDetector signDetector;

    // Setăm parametri mai permisivi pentru ORB
    signDetector.setORBParameters(3, 0.9f); // Minim 3 potriviri, ratio 0.9

    // Încărcăm template-urile
    signDetector.loadTemplates(templatePaths);

    // Creăm o clasă derivată specială pentru a accesa doar metoda ORB
    class ORBOnlyDetector : public SignDetector {
    public:
        using SignDetector::SignDetector;

        std::vector<cv::Rect> detectORBOnly(const cv::Mat& image) {
            // Accesăm direct metoda protejată pentru ORB
            return detectSignsByFeatures(image);
        }
    };

    ORBOnlyDetector orbDetector;
    orbDetector.setORBParameters(3, 0.9f);
    orbDetector.loadTemplates(templatePaths);

    // Detectăm semnele doar cu ORB
    std::vector<cv::Rect> detectedSigns = orbDetector.detectORBOnly(image);

    // Afișăm rezultatele
    cv::Mat resultImage = image.clone();

    if (detectedSigns.empty()) {
        std::cout << "Nu s-au detectat semne folosind doar ORB!" << std::endl;

        // Afișăm potrivirile pentru diagnosticare
        if (!templatePaths.empty()) {
            std::cout << "Afișăm potrivirile cu primul template pentru diagnosticare..." << std::endl;
            displayORBMatches(image, templatePaths[0]);
        }
    } else {
        for (const auto& signRect : detectedSigns) {
            // Desenăm dreptunghiul
            cv::rectangle(resultImage, signRect, cv::Scalar(0, 0, 255), 2);

            // Adăugăm text
            cv::putText(resultImage, "ORB", cv::Point(signRect.x, signRect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }

        std::cout << "S-au detectat " << detectedSigns.size() << " semne folosind doar ORB!" << std::endl;

        // Afișăm rezultatul
        cv::namedWindow("Detecție doar cu ORB", cv::WINDOW_NORMAL);
        cv::imshow("Detecție doar cu ORB", resultImage);
        cv::waitKey(0);
    }
}

int main() {
    // Directorul cu template-uri
    std::string templateDir = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\templates)";
    std::cout << "Testare detecție ORB cu template-uri din: " << templateDir << std::endl;

    // Verificăm dacă directorul există
    if (!std::filesystem::exists(templateDir)) {
        std::cerr << "Directorul de template-uri nu există!" << std::endl;
        return -1;
    }

    // Încărcăm template-urile
    std::vector<std::string> templatePaths;
    for (const auto& entry : std::filesystem::directory_iterator(templateDir)) {
        if (entry.is_regular_file()) {
            std::string extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
                extension == ".bmp" || extension == ".tiff" || extension == ".webp") {
                templatePaths.push_back(entry.path().string());
                std::cout << "S-a găsit template-ul: " << entry.path().string() << std::endl;
            }
        }
    }

    // Verificăm dacă am găsit template-uri
    if (templatePaths.empty()) {
        std::cerr << "Nu s-au găsit template-uri în director!" << std::endl;
        return -1;
    }

    // Calea către imaginea de test
    std::string testImagePath = R"(C:\Users\flavi\OneDrive\Documents\PI\ProiectPI_ORB\stop.jpeg)";

    // Testăm detecția ORB
    testORBDetection(testImagePath, templatePaths);

    return 0;
}