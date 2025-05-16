#include "shape_detector.h"
#include <iostream>
#include <limits>
#include <queue>

ShapeDetector::ShapeDetector() {
    circularityThreshold = 0.85;
    triangularityThreshold = 0.75;
    rectangularityThreshold = 0.8;
}

std::vector<std::vector<cv::Point>> ShapeDetector::detectContours(const cv::Mat& image) {
    cv::Mat grayImage;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    } else {
        grayImage = image.clone();
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(grayImage, blurredImage, cv::Size(5, 5), 0);

    cv::Mat edges;
    edges = detectEdges(blurredImage);

    return traceContours(edges);
}

cv::Mat ShapeDetector::detectEdges(const cv::Mat& grayImage) {
    cv::Mat edges = cv::Mat::zeros(grayImage.size(), CV_8UC1);

    const uchar* prevRow;
    const uchar* currRow;
    const uchar* nextRow;

    for (int y = 1; y < grayImage.rows - 1; y++) {
        prevRow = grayImage.ptr<uchar>(y-1);
        currRow = grayImage.ptr<uchar>(y);
        nextRow = grayImage.ptr<uchar>(y+1);

        uchar* edgeRow = edges.ptr<uchar>(y);

        for (int x = 1; x < grayImage.cols - 1; x++) {
            int gx =
                -prevRow[x-1] + prevRow[x+1] +
                -2*currRow[x-1] + 2*currRow[x+1] +
                -nextRow[x-1] + nextRow[x+1];

            int gy =
                -prevRow[x-1] - 2*prevRow[x] - prevRow[x+1] +
                nextRow[x-1] + 2*nextRow[x] + nextRow[x+1];

            int magnitude = abs(gx) + abs(gy);

            edgeRow[x] = magnitude > 100 ? 255 : 0;
        }
    }

    return edges;
}

std::vector<std::vector<cv::Point>> ShapeDetector::traceContours(const cv::Mat& edges) {
    std::vector<std::vector<cv::Point>> contours;

    cv::Mat visited = cv::Mat::zeros(edges.size(), CV_8UC1);

    for (int y = 1; y < edges.rows - 1; y += 2) {
        const uchar* edgeRow = edges.ptr<uchar>(y);
        uchar* visitedRow = visited.ptr<uchar>(y);

        for (int x = 1; x < edges.cols - 1; x += 2) {
            if (edgeRow[x] == 255 && visitedRow[x] == 0) {
                std::vector<cv::Point> points;
                points.reserve(100);

                cv::Point start(x, y);
                points.push_back(start);
                visitedRow[x] = 255;

                const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
                const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

                int dir = 7;
                cv::Point current = start;

                bool foundNext = true;
                while (foundNext && points.size() < 10000) {
                    foundNext = false;

                    for (int i = 0; i < 8; i++) {
                        int newDir = (dir + 6 + i) % 8;
                        int nx = current.x + dx[newDir];
                        int ny = current.y + dy[newDir];

                        if (nx >= 0 && nx < edges.cols && ny >= 0 && ny < edges.rows) {
                            if (edges.at<uchar>(ny, nx) == 255 && visited.at<uchar>(ny, nx) == 0) {
                                current.x = nx;
                                current.y = ny;
                                dir = newDir;

                                points.push_back(current);
                                visited.at<uchar>(ny, nx) = 255;
                                foundNext = true;
                                break;
                            }
                        }
                    }

                    if (points.size() > 3) {
                        cv::Point first = points[0];
                        cv::Point second = points[1];
                        cv::Point last = points[points.size() - 1];
                        cv::Point secondLast = points[points.size() - 2];

                        if (first == secondLast && second == last) {
                            break;
                            }
                    }

                }

                if (points.size() > 10) {
                    contours.push_back(std::move(points));
                }
            }
        }
    }

    return contours;
}


std::vector<std::pair<ShapeType, std::vector<cv::Point>>> ShapeDetector::detectShapes(const std::vector<std::vector<cv::Point>>& contours) {

    std::vector<std::pair<ShapeType, std::vector<cv::Point>>> shapes;

    for (const auto& contour : contours) {
        ShapeType shapeType = detectShape(contour);
        shapes.push_back(std::make_pair(shapeType, contour));
    }

    return shapes;
}

ShapeType ShapeDetector::detectShape(const std::vector<cv::Point>& contour) {
    double circularityScore = 0.0;
    double triangularityScore = 0.0;
    double rectangularityScore = 0.0;

    bool isCirc = isCircle(contour, circularityScore);
    bool isTri = isTriangle(contour, triangularityScore);
    bool isRect = isRectangle(contour, rectangularityScore);

    if (isCirc && circularityScore > triangularityScore && circularityScore > rectangularityScore) {
        return CIRCLE;
    }
    if (isTri && triangularityScore > circularityScore && triangularityScore > rectangularityScore) {
        return TRIANGLE;
    }
    if (isRect && rectangularityScore > circularityScore && rectangularityScore > triangularityScore) {
        return RECTANGLE;
    }

    return UNKNOWN;
}

bool ShapeDetector::isCircle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateCircularity(contour);
    return similarityScore > circularityThreshold;
}

bool ShapeDetector::isTriangle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateTriangularity(contour);
    return similarityScore > triangularityThreshold;
}

bool ShapeDetector::isRectangle(const std::vector<cv::Point>& contour, double& similarityScore) {
    similarityScore = calculateRectangularity(contour);
    return similarityScore > rectangularityThreshold;
}

double ShapeDetector::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = 0.0;
    double perimeter = 0.0;

    int n = contour.size();
    if (n < 3) return 0.0;

    int i, j;
    for (i = 0, j = n - 1; i < n; j = i++) {
        area += (double)(contour[j].x + contour[i].x) * (contour[j].y - contour[i].y);

        double dx = contour[i].x - contour[j].x;
        double dy = contour[i].y - contour[j].y;
        perimeter += std::sqrt(dx*dx + dy*dy);
    }

    area = std::abs(area) * 0.5;

    if (perimeter <= 0) {
        return 0.0;
    }

    double circularity = (4 * M_PI * area) / (perimeter * perimeter);

    return circularity;
}

double ShapeDetector::calculateTriangularity(const std::vector<cv::Point>& contour) {
    double epsilon = 0.05 * calculatePerimeter(contour);
    std::vector<cv::Point> approx = approximatePolygon(contour, epsilon);

    if (approx.size() != 3) {
        return 0.0;
    }

    double contourArea = calculateArea(contour);

    const cv::Point& p1 = approx[0];
    const cv::Point& p2 = approx[1];
    const cv::Point& p3 = approx[2];

    double a = std::sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
    double b = std::sqrt((p2.x - p3.x) * (p2.x - p3.x) + (p2.y - p3.y) * (p2.y - p3.y));
    double c = std::sqrt((p3.x - p1.x) * (p3.x - p1.x) + (p3.y - p1.y) * (p3.y - p1.y));

    double s = (a + b + c) / 2;
    double triangleArea = std::sqrt(s * (s - a) * (s - b) * (s - c));

    if (triangleArea <= 0) {
        return 0.0;
    }

    return contourArea / triangleArea;
}

double ShapeDetector::calculateRectangularity(const std::vector<cv::Point>& contour) {
    double epsilon = 0.05 * calculatePerimeter(contour);
    std::vector<cv::Point> approx = approximatePolygon(contour, epsilon);

    if (approx.size() != 4) {
        return 0.0;
    }

    double contourArea = calculateArea(contour);

    int minX = approx[0].x, minY = approx[0].y;
    int maxX = approx[0].x, maxY = approx[0].y;

    for (int i = 1; i < 4; i++) {
        minX = std::min(minX, approx[i].x);
        minY = std::min(minY, approx[i].y);
        maxX = std::max(maxX, approx[i].x);
        maxY = std::max(maxY, approx[i].y);
    }

    double rectArea = (maxX - minX) * (maxY - minY);

    if (rectArea <= 0) {
        return 0.0;
    }

    return contourArea / rectArea;
}

double ShapeDetector::calculateArea(const std::vector<cv::Point>& contour) {
    int n = contour.size();

    if (n < 3) return 0.0;

    double area = 0.0;

    int i, j;
    for (i = 0, j = n - 1; i < n; j = i++) {
        area += (double)(contour[j].x + contour[i].x) * (contour[j].y - contour[i].y);
    }

    return std::abs(area) * 0.5;
}

double ShapeDetector::calculatePerimeter(const std::vector<cv::Point>& contour) {
    int n = contour.size();

    if (n < 2) return 0.0;

    double perimeter = 0.0;

    int i, j;
    for (i = 0, j = n - 1; i < n; j = i++) {
        double dx = contour[i].x - contour[j].x;
        double dy = contour[i].y - contour[j].y;
        perimeter += std::sqrt(dx*dx + dy*dy);
    }

    return perimeter;
}

std::vector<cv::Point> ShapeDetector::approximatePolygon(const std::vector<cv::Point>& contour, double epsilon) {

    if (contour.size() <= 2) {
        return contour;
    }

    std::vector<cv::Point> result;
    result.reserve(contour.size() / 5 + 2);

    result.push_back(contour.front());

    struct DPSegment {
        int start;
        int end;
        DPSegment(int s, int e) : start(s), end(e) {}
    };

    std::vector<DPSegment> stack;
    stack.reserve(20);

    stack.push_back(DPSegment(0, contour.size() - 1));

    while (!stack.empty()) {
        DPSegment seg = stack.back();
        stack.pop_back();

        int start = seg.start;
        int end = seg.end;

        if (end - start <= 1) {
            continue;
        }

        double maxDist = 0;
        int maxIndex = start;

        const cv::Point& startPt = contour[start];
        const cv::Point& endPt = contour[end];

        for (int i = start + 1; i < end; i++) {
            double dist = perpendicularDistance(contour[i], startPt, endPt);
            if (dist > maxDist) {
                maxDist = dist;
                maxIndex = i;
            }
        }

        if (maxDist > epsilon) {
            stack.push_back(DPSegment(maxIndex, end));
            stack.push_back(DPSegment(start, maxIndex));
            result.push_back(contour[maxIndex]);
        }
    }

    if (contour.front() != contour.back()) {
        result.push_back(contour.back());
    }

    return result;
}

double ShapeDetector::perpendicularDistance(const cv::Point& point, const cv::Point& lineStart, const cv::Point& lineEnd) {

    double dx = lineEnd.x - lineStart.x;
    double dy = lineEnd.y - lineStart.y;

    if (dx == 0 && dy == 0) {
        double px = point.x - lineStart.x;
        double py = point.y - lineStart.y;
        return std::sqrt(px*px + py*py);
    }

    double areaDouble = std::abs(dx * (lineStart.y - point.y) - dy * (lineStart.x - point.x));

    double length = std::sqrt(dx*dx + dy*dy);

    return areaDouble / length;
}