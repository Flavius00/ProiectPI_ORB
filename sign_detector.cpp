#include "sign_detector.h"
#include <iostream>

SignDetector::SignDetector() : orbDetector(1000, 1.2f, 8) {
    minMatchesRequired = 5;
    matchDistanceRatio = 0.8f;
}

std::vector<cv::Rect> SignDetector::detectSigns(const cv::Mat& image) {

    std::vector<cv::Rect> shapeDetections = detectSignsByShape(image);
    std::vector<cv::Rect> featureDetections = detectSignsByFeatures(image);

    std::cout << "Detectii bazate pe forma: " << shapeDetections.size() << std::endl;
    std::cout << "Detectii bazate pe ORB: " << featureDetections.size() << std::endl;

    std::vector<cv::Rect> allDetections;
    allDetections.insert(allDetections.end(), shapeDetections.begin(), shapeDetections.end());
    allDetections.insert(allDetections.end(), featureDetections.begin(), featureDetections.end());

    return filterOverlappingDetections(allDetections);
}

void SignDetector::loadTemplates(const std::vector<std::string>& templatePaths) {
    templates.clear();

    for (const auto& path : templatePaths) {
        cv::Mat templateImage = cv::imread(path);

        if (templateImage.empty()) {
            std::cerr << "Eroare: Nu s-a putut încărca template-ul: " << path << std::endl;
            continue;
        }

        cv::Mat processedTemplate;
        preprocessTemplate(templateImage, processedTemplate);

        cv::Mat descriptors;
        std::vector<KeyPoint> keypoints = orbDetector.detectAndCompute(processedTemplate, descriptors);

        if (keypoints.size() < 10) {
            std::cerr << "Avertisment: Template-ul are prea puține puncte cheie: " << path << std::endl;
        }

        std::vector<std::vector<cv::Point>> contours = shapeDetector.detectContours(templateImage);
        ShapeType shape = UNKNOWN;

        if (!contours.empty()) {
            auto maxContour = std::max_element(contours.begin(), contours.end(),
                                               [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                                   return cv::contourArea(a) < cv::contourArea(b);
                                               });

            shape = shapeDetector.detectShape(*maxContour);
        }

        SignTemplate signTemplate;
        signTemplate.image = processedTemplate;
        signTemplate.keypoints = keypoints;
        signTemplate.descriptors = descriptors;
        signTemplate.shape = shape;
        signTemplate.originalPath = path;

        templates.push_back(signTemplate);
        std::cout << "Template incarcat: " << path << " cu " << keypoints.size() << " puncte cheie" << std::endl;
    }
}

void SignDetector::preprocessTemplate(const cv::Mat& input, cv::Mat& output) {
    cv::Mat temp;

    if (input.rows > 300 || input.cols > 300) {
        double scale = 300.0 / std::max(input.rows, input.cols);
        cv::resize(input, temp, cv::Size(), scale, scale);
    } else {
        temp = input.clone();
    }

    if (temp.channels() == 3) {
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
    }

    cv::equalizeHist(temp, temp);

    cv::GaussianBlur(temp, output, cv::Size(3, 3), 0);
}

std::vector<cv::Rect> SignDetector::detectSignsByFeatures(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    if (templates.empty()) {
        std::cerr << "Atenție: Nu există template-uri încărcate pentru detecția cu ORB!" << std::endl;
        return detections;
    }

    cv::Mat processedImage;
    preprocessImage(image, processedImage);

    cv::Mat descriptors;
    std::vector<KeyPoint> keypoints = orbDetector.detectAndCompute(processedImage, descriptors);

    if (keypoints.size() < 10) {
        std::cerr << "Avertisment: Imaginea de intrare are prea puține puncte cheie pentru detecția cu ORB!" << std::endl;
        return detections;
    }

    for (const auto& signTemplate : templates) {
        if (signTemplate.descriptors.empty()) {
            continue;
        }

        std::vector<std::pair<int, int>> matches = orbDetector.match(
                descriptors, signTemplate.descriptors, matchDistanceRatio);

        std::cout << "Template: " << signTemplate.originalPath
                  << " - potriviri: " << matches.size() << "/" << minMatchesRequired << std::endl;

        if (matches.size() >= minMatchesRequired) {
            std::vector<cv::Point2f> srcPoints, dstPoints;

            for (const auto& match : matches) {
                srcPoints.push_back(cv::Point2f(keypoints[match.first].x, keypoints[match.first].y));
                dstPoints.push_back(cv::Point2f(signTemplate.keypoints[match.second].x,
                                                signTemplate.keypoints[match.second].y));
            }

            std::vector<uchar> inliersMask;
            cv::Mat H = cv::findHomography(dstPoints, srcPoints, cv::RANSAC, 5.0, inliersMask);
            std::cout << "Homografia" << H << std::endl;
            if (!H.empty()) {
                std::cout << "am intrat aici" << std::endl;
                int numInliers = 0;
                for (uchar mask : inliersMask) {
                    if (mask) numInliers++;
                }

                std::cout << "numInliers: " << numInliers << std::endl;
                std::cout << "numMatches: " << minMatchesRequired * 0.7 << std::endl;

                if (numInliers >= minMatchesRequired * 0.7) {
                    std::vector<cv::Point2f> templateCorners(4);
                    templateCorners[0] = cv::Point2f(0, 0);
                    templateCorners[1] = cv::Point2f(signTemplate.image.cols, 0);
                    templateCorners[2] = cv::Point2f(signTemplate.image.cols, signTemplate.image.rows);
                    templateCorners[3] = cv::Point2f(0, signTemplate.image.rows);

                    std::vector<cv::Point2f> sceneCorners(4);
                    cv::perspectiveTransform(templateCorners, sceneCorners, H);

                    int minX = std::numeric_limits<int>::max();
                    int minY = std::numeric_limits<int>::max();
                    int maxX = std::numeric_limits<int>::min();
                    int maxY = std::numeric_limits<int>::min();

                    for (const auto& corner : sceneCorners) {
                        minX = std::min(minX, static_cast<int>(corner.x));
                        minY = std::min(minY, static_cast<int>(corner.y));
                        maxX = std::max(maxX, static_cast<int>(corner.x));
                        maxY = std::max(maxY, static_cast<int>(corner.y));
                    }
                    std::cout << "minX: " << minX << " minY: " << minY << " maxX: " << maxX << " maxY: " << maxY << std::endl;
                    if (minX < maxX && minY < maxY &&
                        minX >= 0 && minY >= 0 &&
                        maxX < image.cols && maxY < image.rows) {

                        int width = maxX - minX;
                        int height = maxY - minY;
                        float aspectRatio = static_cast<float>(width) / height;

                        if (aspectRatio >= 0.5 && aspectRatio <= 2.0 &&
                            width >= 20 && height >= 20) {

                            cv::Rect boundingRect(minX, minY, width, height);

                            int expansion = std::min(boundingRect.width, boundingRect.height) / 10;
                            boundingRect.x = std::max(0, boundingRect.x - expansion);
                            boundingRect.y = std::max(0, boundingRect.y - expansion);
                            boundingRect.width = std::min(image.cols - boundingRect.x, boundingRect.width + 2 * expansion);
                            boundingRect.height = std::min(image.rows - boundingRect.y, boundingRect.height + 2 * expansion);

                            detections.push_back(boundingRect);
                            std::cout << "Detecție ORB reușită cu template: " << signTemplate.originalPath << std::endl;
                        }
                    }
                }
            }
        }
    }

    return detections;
}

void SignDetector::preprocessImage(const cv::Mat& input, cv::Mat& output) {

    if (input.channels() == 3) {
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
    } else {
        output = input.clone();
    }

    cv::equalizeHist(output, output);

    cv::GaussianBlur(output, output, cv::Size(3, 3), 0);
}

std::vector<cv::Rect> SignDetector::detectSignsByShape(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    std::vector<std::vector<cv::Point>> contours = shapeDetector.detectContours(image);

    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> shapes =
        shapeDetector.detectShapes(contours);

    for (const auto& shape : shapes) {
        ShapeType type = shape.first;
        const std::vector<cv::Point>& contour = shape.second;

        if (type == CIRCLE || type == TRIANGLE || type == RECTANGLE) {
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