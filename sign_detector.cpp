#include "sign_detector.h"
#include <iostream>

SignDetector::SignDetector() : orbDetector(1000, 1.2f, 8) { // Am crescut numărul de features
    // Inițializare parametri mai permisivi
    minMatchesRequired = 5;  // Reducem pragul de potriviri (era 10 implicit)
    matchDistanceRatio = 0.8f; // Facem mai permisiv raportul de potrivire (era 0.75)
}

std::vector<cv::Rect> SignDetector::detectSigns(const cv::Mat& image) {
    // Dacă vrei să testezi doar detecția bazată pe ORB, decomentează linia de mai jos
    // return detectSignsByFeatures(image);

    // Detectăm semnele folosind toate metodele
    std::vector<cv::Rect> colorDetections = detectSignsByColor(image);
    std::vector<cv::Rect> shapeDetections = detectSignsByShape(image);
    std::vector<cv::Rect> featureDetections = detectSignsByFeatures(image);

    // Afișăm câte detecții a găsit fiecare metodă (pentru debugging)
    std::cout << "Detecții bazate pe culoare: " << colorDetections.size() << std::endl;
    std::cout << "Detecții bazate pe formă: " << shapeDetections.size() << std::endl;
    std::cout << "Detecții bazate pe ORB: " << featureDetections.size() << std::endl;

    // Combinăm toate detecțiile
    std::vector<cv::Rect> allDetections;
    allDetections.insert(allDetections.end(), colorDetections.begin(), colorDetections.end());
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

        // Preprocesăm imaginea template pentru mai multe detalii
        cv::Mat processedTemplate;
        preprocessTemplate(templateImage, processedTemplate);

        // Detectăm caracteristicile ORB pentru template
        cv::Mat descriptors;
        std::vector<KeyPoint> keypoints = orbDetector.detectAndCompute(processedTemplate, descriptors);

        // Verificăm dacă avem suficiente puncte cheie
        if (keypoints.size() < 10) {
            std::cerr << "Avertisment: Template-ul are prea puține puncte cheie: " << path << std::endl;
            // Continuăm oricum, poate fi util în combinație cu alte metode
        }

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

        // Adăugăm template-ul la listă
        SignTemplate signTemplate;
        signTemplate.image = processedTemplate;
        signTemplate.keypoints = keypoints;
        signTemplate.descriptors = descriptors;
        signTemplate.shape = shape;
        signTemplate.originalPath = path;  // Salvăm și calea pentru debugging

        templates.push_back(signTemplate);
        std::cout << "Template încărcat: " << path << " cu " << keypoints.size() << " puncte cheie" << std::endl;
    }
}

void SignDetector::preprocessTemplate(const cv::Mat& input, cv::Mat& output) {
    // Aplicăm prelucrări pentru a îmbunătăți extragerea caracteristicilor
    cv::Mat temp;

    // Redimensionăm template-ul pentru a asigura o dimensiune rezonabilă
    if (input.rows > 300 || input.cols > 300) {
        double scale = 300.0 / std::max(input.rows, input.cols);
        cv::resize(input, temp, cv::Size(), scale, scale);
    } else {
        temp = input.clone();
    }

    // Convertim la grayscale dacă e color
    if (temp.channels() == 3) {
        cv::cvtColor(temp, temp, cv::COLOR_BGR2GRAY);
    }

    // Îmbunătățim contrastul pentru mai multe detalii
    cv::equalizeHist(temp, temp);

    // Aplicăm un filtru Gaussian pentru a reduce zgomotul, păstrând detaliile
    cv::GaussianBlur(temp, output, cv::Size(3, 3), 0);
}

std::vector<cv::Rect> SignDetector::detectSignsByFeatures(const cv::Mat& image) {
    std::vector<cv::Rect> detections;

    // Dacă nu avem template-uri încărcate, returnăm o listă goală
    if (templates.empty()) {
        std::cerr << "Atenție: Nu există template-uri încărcate pentru detecția cu ORB!" << std::endl;
        return detections;
    }

    // Preprocesăm imaginea de intrare
    cv::Mat processedImage;
    preprocessImage(image, processedImage);

    // Detectăm caracteristicile ORB în imaginea de intrare
    cv::Mat descriptors;
    std::vector<KeyPoint> keypoints = orbDetector.detectAndCompute(processedImage, descriptors);

    if (keypoints.size() < 10) {
        std::cerr << "Avertisment: Imaginea de intrare are prea puține puncte cheie pentru detecția cu ORB!" << std::endl;
        return detections;
    }

    // Pentru fiecare template
    for (const auto& signTemplate : templates) {
        // Verificăm dacă template-ul are descriptori
        if (signTemplate.descriptors.empty()) {
            continue;
        }

        // Potrivim descriptorii
        std::vector<std::pair<int, int>> matches = orbDetector.match(
                descriptors, signTemplate.descriptors, matchDistanceRatio);

        // Afișăm numărul de potriviri pentru debugging
        std::cout << "Template: " << signTemplate.originalPath
                  << " - potriviri: " << matches.size() << "/" << minMatchesRequired << std::endl;

        // Dacă găsim suficiente potriviri
        if (matches.size() >= minMatchesRequired) {
            // Colectăm punctele potrivite
            std::vector<cv::Point2f> srcPoints, dstPoints;

            for (const auto& match : matches) {
                srcPoints.push_back(cv::Point2f(keypoints[match.first].x, keypoints[match.first].y));
                dstPoints.push_back(cv::Point2f(signTemplate.keypoints[match.second].x,
                                                signTemplate.keypoints[match.second].y));
            }

            // Calculăm transformarea omografică cu RANSAC mai permisiv
            std::vector<uchar> inliersMask;
            cv::Mat H = cv::findHomography(dstPoints, srcPoints, cv::RANSAC, 5.0, inliersMask);

            if (!H.empty()) {
                // Numărăm inliers după RANSAC
                int numInliers = 0;
                for (uchar mask : inliersMask) {
                    if (mask) numInliers++;
                }

                // Continuăm doar dacă avem suficienți inliers
                if (numInliers >= minMatchesRequired * 0.7) {
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

                        // Verifică și raportul de aspect pentru a evita detecții irelevante
                        int width = maxX - minX;
                        int height = maxY - minY;
                        float aspectRatio = static_cast<float>(width) / height;

                        if (aspectRatio >= 0.5 && aspectRatio <= 2.0 &&
                            width >= 20 && height >= 20) {  // Dimensiuni minime rezonabile

                            cv::Rect boundingRect(minX, minY, width, height);

                            // Extindem dreptunghiul pentru a prinde întregul semn
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
    // Preprocesăm imaginea de intrare pentru detecția ORB

    // Convertim la grayscale pentru ORB
    if (input.channels() == 3) {
        cv::cvtColor(input, output, cv::COLOR_BGR2GRAY);
    } else {
        output = input.clone();
    }

    // Îmbunătățim contrastul pentru a evidenția caracteristicile
    cv::equalizeHist(output, output);

    // Aplicăm un filtru Gaussian pentru a reduce zgomotul
    cv::GaussianBlur(output, output, cv::Size(3, 3), 0);
}