#include "orb_detector.h"
#include <random>
#include <algorithm>

ORBDetector::ORBDetector(int _nfeatures, float _scaleFactor, int _nlevels)
        : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels) {
}

std::vector<KeyPoint> ORBDetector::detect(const cv::Mat& image) {
    // Convertește imaginea la grayscale dacă este necesară
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Detectează colțuri FAST
    std::vector<KeyPoint> keypoints = detectFAST(grayImage);

    // Sortează punctele cheie după răspuns
    std::sort(keypoints.begin(), keypoints.end(),
              [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });

    // Limitează numărul de puncte cheie
    if (keypoints.size() > nfeatures) {
        keypoints.resize(nfeatures);
    }

    // Calculează orientarea pentru punctele cheie
    computeOrientation(grayImage, keypoints);

    return keypoints;
}

cv::Mat ORBDetector::compute(const cv::Mat& image, std::vector<KeyPoint>& keypoints) {
    // Convertește imaginea la grayscale dacă este necesară
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    // Calculează descriptori BRIEF
    return computeBRIEF(grayImage, keypoints);
}

std::vector<KeyPoint> ORBDetector::detectAndCompute(const cv::Mat& image, cv::Mat& descriptors) {
    std::vector<KeyPoint> keypoints = detect(image);
    descriptors = compute(image, keypoints);
    return keypoints;
}

std::vector<KeyPoint> ORBDetector::detectFAST(const cv::Mat& image, int threshold) {
    std::vector<KeyPoint> keypoints;
    const int patternSize = 16; // FAST folosește 16 pixeli într-un cerc

    // Offset pentru verificarea pixelilor în cerc
    static const int offsetsX[patternSize] = {
            0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1
    };
    static const int offsetsY[patternSize] = {
            -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3
    };

    // Aplică detecția FAST
    for (int y = 3; y < image.rows - 3; y++) {
        for (int x = 3; x < image.cols - 3; x++) {
            const uchar centerIntensity = image.at<uchar>(y, x);
            const uchar upperThreshold = centerIntensity + threshold;
            const uchar lowerThreshold = centerIntensity - threshold;

            // Verifică dacă avem un potențial colț
            int countBrighter = 0;
            int countDarker = 0;

            // Verifică pixelii în cerc
            for (int i = 0; i < patternSize; i++) {
                const uchar intensity = image.at<uchar>(y + offsetsY[i], x + offsetsX[i]);

                if (intensity > upperThreshold) countBrighter++;
                else if (intensity < lowerThreshold) countDarker++;
            }

            // Verifică dacă avem 9 pixeli consecutivi care sunt mai luminoși sau mai întunecați
            const int requiredConsecutive = 9;
            bool isCorner = false;

            if (countBrighter >= requiredConsecutive || countDarker >= requiredConsecutive) {
                // Verifică pixelii consecutivi
                for (int i = 0; i < patternSize; i++) {
                    int count = 0;
                    bool brighter = true;

                    // Verifică pixelii consecutivi mai luminoși
                    for (int j = i; j < i + requiredConsecutive; j++) {
                        const int idx = j % patternSize;
                        const uchar intensity = image.at<uchar>(y + offsetsY[idx], x + offsetsX[idx]);

                        if (intensity > upperThreshold) count++;
                        else {
                            brighter = false;
                            break;
                        }
                    }

                    if (brighter && count >= requiredConsecutive) {
                        isCorner = true;
                        break;
                    }

                    count = 0;
                    bool darker = true;

                    // Verifică pixelii consecutivi mai întunecați
                    for (int j = i; j < i + requiredConsecutive; j++) {
                        const int idx = j % patternSize;
                        const uchar intensity = image.at<uchar>(y + offsetsY[idx], x + offsetsX[idx]);

                        if (intensity < lowerThreshold) count++;
                        else {
                            darker = false;
                            break;
                        }
                    }

                    if (darker && count >= requiredConsecutive) {
                        isCorner = true;
                        break;
                    }
                }
            }

            if (isCorner) {
                // Calculăm răspunsul colțului ca diferența absolută între pixelul central și pixelii din cerc
                float cornerResponse = 0.0f;
                for (int i = 0; i < patternSize; i++) {
                    const uchar intensity = image.at<uchar>(y + offsetsY[i], x + offsetsX[i]);
                    cornerResponse += std::abs(static_cast<float>(intensity) - static_cast<float>(centerIntensity));
                }

                keypoints.push_back(KeyPoint(x, y, cornerResponse / patternSize));
            }
        }
    }

    return keypoints;
}

void ORBDetector::computeOrientation(const cv::Mat& image, std::vector<KeyPoint>& keypoints) {
    for (auto& kp : keypoints) {
        const int x = kp.x;
        const int y = kp.y;
        const int radius = 15; // Raza pentru calculul momentelor

        // Calculul momentului
        float m01 = calculateCenteredMoment(image, 0, 1, x, y, radius);
        float m10 = calculateCenteredMoment(image, 1, 0, x, y, radius);

        // Calculul orientării
        kp.angle = std::atan2(m01, m10);

        // Convertirea din radiani la grade
        kp.angle = kp.angle * 180.0f / M_PI;

        // Normalizarea unghiului în intervalul [0, 360)
        if (kp.angle < 0) {
            kp.angle += 360.0f;
        }
    }
}

float ORBDetector::calculateCenteredMoment(const cv::Mat& image, int p, int q, int cx, int cy, int radius) {
    float moment = 0.0f;

    for (int dy = -radius; dy <= radius; dy++) {
        int y = cy + dy;
        if (y < 0 || y >= image.rows) continue;

        for (int dx = -radius; dx <= radius; dx++) {
            int x = cx + dx;
            if (x < 0 || x >= image.cols) continue;

            // Distanța de la centru
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist > radius) continue;

            // Calculul momentului centrat
            float value = static_cast<float>(image.at<uchar>(y, x));
            moment += std::pow(static_cast<float>(dx), p) * std::pow(static_cast<float>(dy), q) * value;
        }
    }

    return moment;
}

std::vector<std::pair<cv::Point, cv::Point>> ORBDetector::generateBriefPattern(int patchSize, int numPairs) {
    std::vector<std::pair<cv::Point, cv::Point>> pattern;

    // Generatorul de numere aleatorii
    std::mt19937 rng(12345); // Seed fix pentru reproducibilitate
    std::uniform_int_distribution<> dist(-patchSize/2, patchSize/2);

    for (int i = 0; i < numPairs; i++) {
        cv::Point p1(dist(rng), dist(rng));
        cv::Point p2(dist(rng), dist(rng));
        pattern.push_back(std::make_pair(p1, p2));
    }

    return pattern;
}

cv::Mat ORBDetector::computeBRIEF(const cv::Mat& image, std::vector<KeyPoint>& keypoints) {
    const int numPairs = 256; // Numărul de perechi pentru descriptorul BRIEF
    const int patchSize = 31; // Dimensiunea patch-ului

    // Generează modelul BRIEF
    std::vector<std::pair<cv::Point, cv::Point>> pattern = generateBriefPattern(patchSize, numPairs);

    // Alocă spațiu pentru descriptori (32 bytes per keypoint)
    cv::Mat descriptors = cv::Mat::zeros(keypoints.size(), 32, CV_8UC1);

    // Pentru fiecare punct cheie
    for (size_t i = 0; i < keypoints.size(); i++) {
        const KeyPoint& kp = keypoints[i];

        // Calculăm matricea de rotație
        float angleRad = kp.angle * M_PI / 180.0f;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);

        // Pentru fiecare pereche din model
        for (int j = 0; j < numPairs; j++) {
            // Obținem perechea de puncte
            const auto& pair = pattern[j];

            // Rotăm punctele conform orientării keypoint-ului
            int x1 = cvRound(pair.first.x * cosA - pair.first.y * sinA) + kp.x;
            int y1 = cvRound(pair.first.x * sinA + pair.first.y * cosA) + kp.y;

            int x2 = cvRound(pair.second.x * cosA - pair.second.y * sinA) + kp.x;
            int y2 = cvRound(pair.second.x * sinA + pair.second.y * cosA) + kp.y;

            // Verificăm dacă punctele sunt în imagine
            if (x1 < 0 || x1 >= image.cols || y1 < 0 || y1 >= image.rows ||
                x2 < 0 || x2 >= image.cols || y2 < 0 || y2 >= image.rows) {
                continue;
            }

            // Comparăm intensitățile
            uchar val1 = image.at<uchar>(y1, x1);
            uchar val2 = image.at<uchar>(y2, x2);

            // Setăm bitul corespunzător
            if (val1 < val2) {
                descriptors.at<uchar>(i, j / 8) |= (1 << (j % 8));
            }
        }
    }

    return descriptors;
}

int ORBDetector::hammingDistance(const cv::Mat& desc1, const cv::Mat& desc2) {
    int distance = 0;

    // Fiecare descriptor este un vector de 32 bytes
    for (int i = 0; i < 32; i++) {
        uchar xor_result = desc1.at<uchar>(0, i) ^ desc2.at<uchar>(0, i);

        // Numărul de biți setați în rezultatul XOR
        while (xor_result) {
            distance += xor_result & 1;
            xor_result >>= 1;
        }
    }

    return distance;
}

std::vector<std::pair<int, int>> ORBDetector::match(const cv::Mat& descriptors1,
                                                    const cv::Mat& descriptors2,
                                                    float threshold) {
    std::vector<std::pair<int, int>> matches;

    const int numDesc1 = descriptors1.rows;
    const int numDesc2 = descriptors2.rows;

    // Pentru fiecare descriptor din primul set
    for (int i = 0; i < numDesc1; i++) {
        int bestDistance = std::numeric_limits<int>::max();
        int secondBestDistance = std::numeric_limits<int>::max();
        int bestIdx = -1;

        // Calculăm distanța față de toți descriptorii din al doilea set
        for (int j = 0; j < numDesc2; j++) {
            int distance = hammingDistance(descriptors1.row(i), descriptors2.row(j));

            if (distance < bestDistance) {
                secondBestDistance = bestDistance;
                bestDistance = distance;
                bestIdx = j;
            } else if (distance < secondBestDistance) {
                secondBestDistance = distance;
            }
        }

        // Aplicăm raportul de verificare a puterii discriminatorii
        if (bestDistance < (float)secondBestDistance * threshold) {
            matches.push_back(std::make_pair(i, bestIdx));
        }
    }

    return matches;
}