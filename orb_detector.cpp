#include "orb_detector.h"
#include <random>
#include <algorithm>

ORBDetector::ORBDetector(int _nfeatures, float _scaleFactor, int _nlevels)
        : nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels) {
}

std::vector<KeyPoint> ORBDetector::detect(const cv::Mat& image) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    std::vector<KeyPoint> keypoints = detectFAST(grayImage);

    std::sort(keypoints.begin(), keypoints.end(),
              [](const KeyPoint& a, const KeyPoint& b) { return a.response > b.response; });

    if (keypoints.size() > nfeatures) {
        keypoints.resize(nfeatures);
    }

    computeOrientation(grayImage, keypoints);

    return keypoints;
}

cv::Mat ORBDetector::compute(const cv::Mat& image, std::vector<KeyPoint>& keypoints) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    return computeBRIEF(grayImage, keypoints);
}

std::vector<KeyPoint> ORBDetector::detectAndCompute(const cv::Mat& image, cv::Mat& descriptors) {
    std::vector<KeyPoint> keypoints = detect(image);
    descriptors = compute(image, keypoints);
    return keypoints;
}

std::vector<KeyPoint> ORBDetector::detectFAST(const cv::Mat& image, int threshold) {
    std::vector<KeyPoint> keypoints;
    const int patternSize = 16;

    static const int offsetsX[patternSize] = {
            0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1
    };
    static const int offsetsY[patternSize] = {
            -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3
    };

    for (int y = 3; y < image.rows - 3; y++) {
        for (int x = 3; x < image.cols - 3; x++) {
            const uchar centerIntensity = image.at<uchar>(y, x);
            const uchar upperThreshold = centerIntensity + threshold;
            const uchar lowerThreshold = centerIntensity - threshold;

            int countBrighter = 0;
            int countDarker = 0;

            for (int i = 0; i < patternSize; i++) {
                const uchar intensity = image.at<uchar>(y + offsetsY[i], x + offsetsX[i]);

                if (intensity > upperThreshold) countBrighter++;
                else if (intensity < lowerThreshold) countDarker++;
            }

            const int requiredConsecutive = 9;
            bool isCorner = false;

            if (countBrighter >= requiredConsecutive || countDarker >= requiredConsecutive) {
                for (int i = 0; i < patternSize; i++) {
                    int count = 0;
                    bool brighter = true;

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
        const int radius = 15;

        float m01 = calculateCenteredMoment(image, 0, 1, x, y, radius);
        float m10 = calculateCenteredMoment(image, 1, 0, x, y, radius);

        kp.angle = std::atan2(m01, m10);

        kp.angle = kp.angle * 180.0f / M_PI;

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

            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist > radius) continue;

            float value = static_cast<float>(image.at<uchar>(y, x));
            moment += std::pow(static_cast<float>(dx), p) * std::pow(static_cast<float>(dy), q) * value;
        }
    }

    return moment;
}

std::vector<std::pair<cv::Point, cv::Point>> ORBDetector::generateBriefPattern(int patchSize, int numPairs) {
    std::vector<std::pair<cv::Point, cv::Point>> pattern;

    std::mt19937 rng(12345);
    std::uniform_int_distribution<> dist(-patchSize/2, patchSize/2);

    for (int i = 0; i < numPairs; i++) {
        cv::Point p1(dist(rng), dist(rng));
        cv::Point p2(dist(rng), dist(rng));
        pattern.push_back(std::make_pair(p1, p2));
    }

    return pattern;
}

cv::Mat ORBDetector::computeBRIEF(const cv::Mat& image, std::vector<KeyPoint>& keypoints) {
    const int numPairs = 256;
    const int patchSize = 31;

    std::vector<std::pair<cv::Point, cv::Point>> pattern = generateBriefPattern(patchSize, numPairs);

    cv::Mat descriptors = cv::Mat::zeros(keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++) {
        const KeyPoint& kp = keypoints[i];

        float angleRad = kp.angle * M_PI / 180.0f;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);

        for (int j = 0; j < numPairs; j++) {
            const auto& pair = pattern[j];

            int x1 = cvRound(pair.first.x * cosA - pair.first.y * sinA) + kp.x;
            int y1 = cvRound(pair.first.x * sinA + pair.first.y * cosA) + kp.y;

            int x2 = cvRound(pair.second.x * cosA - pair.second.y * sinA) + kp.x;
            int y2 = cvRound(pair.second.x * sinA + pair.second.y * cosA) + kp.y;

            if (x1 < 0 || x1 >= image.cols || y1 < 0 || y1 >= image.rows ||
                x2 < 0 || x2 >= image.cols || y2 < 0 || y2 >= image.rows) {
                continue;
            }

            uchar val1 = image.at<uchar>(y1, x1);
            uchar val2 = image.at<uchar>(y2, x2);

            if (val1 < val2) {
                descriptors.at<uchar>(i, j / 8) |= (1 << (j % 8));
            }
        }
    }

    return descriptors;
}

int ORBDetector::hammingDistance(const cv::Mat& desc1, const cv::Mat& desc2) {
    int distance = 0;

    for (int i = 0; i < 32; i++) {
        uchar xor_result = desc1.at<uchar>(0, i) ^ desc2.at<uchar>(0, i);

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

    for (int i = 0; i < numDesc1; i++) {
        int bestDistance = std::numeric_limits<int>::max();
        int secondBestDistance = std::numeric_limits<int>::max();
        int bestIdx = -1;

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

        if (bestDistance < (float)secondBestDistance * threshold) {
            matches.push_back(std::make_pair(i, bestIdx));
        }
    }

    return matches;
}