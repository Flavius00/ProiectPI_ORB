#include "opencv_detector.h"
#include "shape_type.h"

OpenCVShapeDetector::OpenCVShapeDetector() {
    // Inițializarea parametrilor pentru detecția formelor
    circularityThreshold = 0.85;
    triangularityThreshold = 0.75;
    rectangularityThreshold = 0.8;
}

std::vector<std::vector<cv::Point>> OpenCVShapeDetector::detectContours(const cv::Mat& image) {
    // Convertește imaginea la grayscale dacă este necesară
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Aplică un blur pentru a reduce zgomotul
    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

    // Aplică detecția de margini Canny
    cv::Mat edges;
    cv::Canny(blurredImage, edges, 50, 150);

    // Găsește contururile din imaginea cu margini
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Filtrează contururile mici
    std::vector<std::vector<cv::Point>> filteredContours;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > 100) { // Filtrăm contururile foarte mici
            filteredContours.push_back(contour);
        }
    }

    return filteredContours;
}

std::vector<std::pair<ShapeType, std::vector<cv::Point>>> OpenCVShapeDetector::detectShapes(
    const std::vector<std::vector<cv::Point>>& contours) {
    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> shapes;

    for (const auto& contour : contours) {
        ShapeType shapeType = detectShape(contour);
        shapes.push_back(std::make_pair(shapeType, contour));
    }

    return shapes;
}

ShapeType OpenCVShapeDetector::detectShape(const std::vector<cv::Point>& contour) {
    double circularityScore = 0.0;
    double triangularityScore = 0.0;
    double rectangularityScore = 0.0;

    bool isCirc = isCircle(contour, circularityScore);
    bool isTri = isTriangle(contour, triangularityScore);
    bool isRect = isRectangle(contour, rectangularityScore);

    // Determină care formă are scorul de similaritate cel mai mare
    if (isCirc && circularityScore > triangularityScore && circularityScore > rectangularityScore) {
        return ShapeType::CIRCLE;
    } else if (isTri && triangularityScore > circularityScore && triangularityScore > rectangularityScore) {
        return ShapeType::TRIANGLE;
    } else if (isRect && rectangularityScore > circularityScore && rectangularityScore > triangularityScore) {
        return ShapeType::RECTANGLE;
    }

    return ShapeType::UNKNOWN;
}

bool OpenCVShapeDetector::isCircle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateCircularity(contour);
    return similarityScore > circularityThreshold;
}

bool OpenCVShapeDetector::isTriangle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateTriangularity(contour);
    return similarityScore > triangularityThreshold;
}

bool OpenCVShapeDetector::isRectangle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateRectangularity(contour);
    return similarityScore > rectangularityThreshold;
}

double OpenCVShapeDetector::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);

    // Formula pentru circularitate: 4 * pi * area / (perimeter^2)
    // Pentru un cerc perfect, această valoare este 1
    double circularity = (4 * 3.14159265358979323846 * area) / (perimeter * perimeter);

    return circularity;
}

double OpenCVShapeDetector::calculateTriangularity(const std::vector<cv::Point>& contour) {
    // Aproximăm conturul cu un poligon
    std::vector<cv::Point> approx;
    double epsilon = 0.04 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    // Verificăm dacă poligonul are 3 laturi
    if (approx.size() != 3) {
        return 0.0;
    }

    // Calculăm raportul dintre aria conturului și aria triunghiului
    double contourArea = cv::contourArea(contour);
    double triangleArea = cv::contourArea(approx);

    if (triangleArea == 0) {
        return 0.0;
    }

    return contourArea / triangleArea;
}

double OpenCVShapeDetector::calculateRectangularity(const std::vector<cv::Point>& contour) {
    // Aproximăm conturul cu un poligon
    std::vector<cv::Point> approx;
    double epsilon = 0.04 * cv::arcLength(contour, true);
    cv::approxPolyDP(contour, approx, epsilon, true);

    // Verificăm dacă poligonul are 4 laturi
    if (approx.size() != 4) {
        return 0.0;
    }

    // Găsim dreptunghiul cu aria minimă care conține conturul
    cv::RotatedRect boundingRect = cv::minAreaRect(contour);
    double boundingRectArea = boundingRect.size.width * boundingRect.size.height;

    // Calculăm aria conturului
    double contourArea = cv::contourArea(contour);

    if (boundingRectArea == 0) {
        return 0.0;
    }

    // Raportul dintre aria conturului și aria dreptunghiului este o măsură a "rectangularității"
    return contourArea / boundingRectArea;
}