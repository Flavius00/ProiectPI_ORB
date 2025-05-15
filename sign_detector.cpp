#include "sign_detector.h"

SignDetector::SignDetector() : orbDetector(500, 1.2f, 8) {
    // Inițializare detectori
}

std::vector<cv::Rect> SignDetector::detectSigns(const cv::Mat& image) {
    // Detectăm semnele folosind toate metodele
    //std::vector<cv::Rect> colorDetections = detectSignsByColor(image);
    std::vector<cv::Rect> shapeDetections = detectSignsByShape(image);
    std::vector<cv::Rect> featureDetections = detectSignsByFeatures(image);

    // Combinăm toate detecțiile
    std::vector<cv::Rect> allDetections;
    //allDetections.insert(allDetections.end(), colorDetections.begin(), colorDetections.end());
    allDetections.insert(allDetections.end(), shapeDetections.begin(), shapeDetections.end());
    allDetections.insert(allDetections.end(), featureDetections.begin(), featureDetections.end());

    // Filtrăm detecțiile suprapuse
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

        // Detectăm caracteristicile ORB pentru template
        cv::Mat descriptors;
        std::vector<KeyPoint> keypoints = orbDetector.detectAndCompute(templateImage, descriptors);

        // Detectăm forma template-ului
        std::vector<std::vector<cv::Point>> contours = shapeDetector.detectContours(templateImage);
        ShapeType shape = UNKNOWN;

        if (!contours.empty()) {
            // Alegem conturul cu cea mai mare arie
            auto maxContour = std::max_element(contours.begin(), contours.end(),
                                               [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                                                   return cv::contourArea(a) < cv::contourArea(b);
                                               });

            shape = shapeDetector.detectShape(*maxContour);
        }

        // Adăugăm template-ul la lista
        SignTemplate signTemplate;
        signTemplate.image = templateImage;
        signTemplate.keypoints = keypoints;
        signTemplate.descriptors = descriptors;
        signTemplate.shape = shape;

        templates.push_back(signTemplate);
    }
}

std::vector<cv::Rect> SignDetector::detectSignsByColor(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    // Convertim imaginea în spațiul de culoare HSV
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Definim intervalele de culoare pentru detecția culorilor comune ale semnelor de circulație

    // Roșu (două intervale deoarece roșul este la ambele capete ale intervalului H)
    cv::Mat redMask1, redMask2, redMask;
    cv::inRange(hsvImage, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), redMask1);
    cv::inRange(hsvImage, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), redMask2);
    cv::bitwise_or(redMask1, redMask2, redMask);

    // Albastru
    cv::Mat blueMask;
    cv::inRange(hsvImage, cv::Scalar(100, 100, 100), cv::Scalar(130, 255, 255), blueMask);

    // Galben
    cv::Mat yellowMask;
    cv::inRange(hsvImage, cv::Scalar(20, 100, 100), cv::Scalar(30, 255, 255), yellowMask);

    // Combinăm toate măștile
    cv::Mat combinedMask;
    cv::bitwise_or(redMask, blueMask, combinedMask);
    cv::bitwise_or(combinedMask, yellowMask, combinedMask);

    // Aplicăm operații morfologice pentru a îmbunătăți măștile
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, kernel);

    // Găsim contururile
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(combinedMask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Pentru fiecare contur, creăm un dreptunghi de încadrare
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);

        // Filtrăm contururile mici
        if (area > 100) {
            cv::Rect boundingRect = cv::boundingRect(contour);

            // Extindem dreptunghiul pentru a prinde întregul semn
            int expansion = std::min(boundingRect.width, boundingRect.height) / 10;
            boundingRect.x = std::max(0, boundingRect.x - expansion);
            boundingRect.y = std::max(0, boundingRect.y - expansion);
            boundingRect.width = std::min(image.cols - boundingRect.x, boundingRect.width + 2 * expansion);
            boundingRect.height = std::min(image.rows - boundingRect.y, boundingRect.height + 2 * expansion);

            detections.push_back(boundingRect);
        }
    }

    return detections;
}

std::vector<cv::Rect> SignDetector::detectSignsByShape(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    // Găsim contururile din imagine
    std::vector<std::vector<cv::Point>> contours = shapeDetector.detectContours(image);

    // Detectăm formele
    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> shapes = shapeDetector.detectShapes(contours);

    // Pentru fiecare formă detectată
    for (const auto& shape : shapes) {
        ShapeType shapeType = shape.first;
        const std::vector<cv::Point>& contour = shape.second;

        // Verificăm dacă este o formă relevantă pentru semnele de circulație
        if (shapeType == CIRCLE || shapeType == TRIANGLE || shapeType == RECTANGLE) {
            cv::Rect boundingRect = cv::boundingRect(contour);

            // Calculăm raportul de aspect și verificăm dacă este apropiat de 1 (semn pătratic)
            double aspectRatio = static_cast<double>(boundingRect.width) / boundingRect.height;

            if (aspectRatio >= 0.7 && aspectRatio <= 1.3) {
                // Extindem dreptunghiul pentru a prinde întregul semn
                int expansion = std::min(boundingRect.width, boundingRect.height) / 10;
                boundingRect.x = std::max(0, boundingRect.x - expansion);
                boundingRect.y = std::max(0, boundingRect.y - expansion);
                boundingRect.width = std::min(image.cols - boundingRect.x, boundingRect.width + 2 * expansion);
                boundingRect.height = std::min(image.rows - boundingRect.y, boundingRect.height + 2 * expansion);

                detections.push_back(boundingRect);
            }
        }
    }

    return detections;
}

std::vector<cv::Rect> SignDetector::detectSignsByFeatures(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    // Dacă nu avem template-uri încărcate, returnăm o listă goală
    if (templates.empty()) {
        return detections;
    }

    // Detectăm caracteristicile ORB în imaginea de intrare
    cv::Mat descriptors;
    std::vector<KeyPoint> keypoints = orbDetector.detectAndCompute(image, descriptors);

    // Pentru fiecare template
    for (const auto& signTemplate : templates) {
        // Potrivim descriptorii
        std::vector<std::pair<int, int>> matches = orbDetector.match(descriptors, signTemplate.descriptors, 0.75f);

        // Dacă găsim suficiente potriviri
        if (matches.size() >= 10) {
            // Colectăm punctele potrivite
            std::vector<cv::Point2f> srcPoints, dstPoints;

            for (const auto& match : matches) {
                srcPoints.push_back(cv::Point2f(keypoints[match.first].x, keypoints[match.first].y));
                dstPoints.push_back(cv::Point2f(signTemplate.keypoints[match.second].x, signTemplate.keypoints[match.second].y));
            }

            // Calculăm transformarea omografică
            cv::Mat H = cv::findHomography(dstPoints, srcPoints, cv::RANSAC);

            if (!H.empty()) {
                // Transformăm colțurile template-ului pentru a găsi semnul în imagine
                std::vector<cv::Point2f> templateCorners(4);
                templateCorners[0] = cv::Point2f(0, 0);
                templateCorners[1] = cv::Point2f(signTemplate.image.cols, 0);
                templateCorners[2] = cv::Point2f(signTemplate.image.cols, signTemplate.image.rows);
                templateCorners[3] = cv::Point2f(0, signTemplate.image.rows);

                std::vector<cv::Point2f> sceneCorners(4);
                cv::perspectiveTransform(templateCorners, sceneCorners, H);

                // Găsim dreptunghiul de încadrare al semnului detectat
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

                // Verificăm dacă dreptunghiul este valid și în limitele imaginii
                if (minX < maxX && minY < maxY &&
                    minX >= 0 && minY >= 0 &&
                    maxX < image.cols && maxY < image.rows) {

                    cv::Rect boundingRect(minX, minY, maxX - minX, maxY - minY);

                    // Extindem dreptunghiul pentru a prinde întregul semn
                    int expansion = std::min(boundingRect.width, boundingRect.height) / 10;
                    boundingRect.x = std::max(0, boundingRect.x - expansion);
                    boundingRect.y = std::max(0, boundingRect.y - expansion);
                    boundingRect.width = std::min(image.cols - boundingRect.x, boundingRect.width + 2 * expansion);
                    boundingRect.height = std::min(image.rows - boundingRect.y, boundingRect.height + 2 * expansion);

                    detections.push_back(boundingRect);
                }
            }
        }
    }

    return detections;
}

std::vector<cv::Rect> SignDetector::filterOverlappingDetections(const std::vector<cv::Rect>& detections, float overlapThreshold) {
    if (detections.empty()) {
        return detections;
    }

    std::vector<cv::Rect> filteredDetections;
    std::vector<bool> isOverlapped(detections.size(), false);

    // Pentru fiecare detecție
    for (size_t i = 0; i < detections.size(); i++) {
        // Dacă detecția a fost deja marcată ca fiind suprapusă, o ignorăm
        if (isOverlapped[i]) {
            continue;
        }

        cv::Rect currentRect = detections[i];

        // Verificăm suprapunerea cu celelalte detecții
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (isOverlapped[j]) {
                continue;
            }

            cv::Rect otherRect = detections[j];

            // Calculăm IoU
            float iou = calculateIoU(currentRect, otherRect);

            // Dacă IoU este peste prag, combinăm detecțiile
            if (iou > overlapThreshold) {
                // Combinăm detecțiile într-un dreptunghi care le cuprinde pe ambele
                currentRect = currentRect | otherRect;
                isOverlapped[j] = true;
            }
        }

        filteredDetections.push_back(currentRect);
    }

    return filteredDetections;
}

float SignDetector::calculateIoU(const cv::Rect& rect1, const cv::Rect& rect2) {
    // Calculăm intersecția
    cv::Rect intersection = rect1 & rect2;

    // Dacă nu există intersecție, IoU este 0
    if (intersection.width <= 0 || intersection.height <= 0) {
        return 0.0f;
    }

    // Calculăm aria intersecției
    float intersectionArea = intersection.width * intersection.height;

    // Calculăm aria uniunii
    float unionArea = rect1.width * rect1.height + rect2.width * rect2.height - intersectionArea;

    // Calculăm IoU
    return intersectionArea / unionArea;
}