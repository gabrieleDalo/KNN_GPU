#ifndef UTILS_H
#define UTILS_H

struct Neighbor {
    int index;
    float distance;
};

void printTime(double time);
float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b);
float calculatePrecision(const std::vector<std::vector<Neighbor>>& knNeighbors1, const std::vector<std::vector<Neighbor>>& knNeighbors2);
float compareGroundtruth(const int dbSize, const std::vector<std::vector<int>>& tests, const int k, std::vector<std::vector<Neighbor>>& exactNeighbors);
std::pair<double, double> findKNN(const std::vector<std::vector<float>>& database, const std::vector<std::vector<float>>& queries, const int k, std::vector<std::vector<Neighbor>>& knNeighbors);
std::pair<double, double> findKNN_minHeap(const std::vector<std::vector<float>>& database, const std::vector<std::vector<float>>& queries, const int k, std::vector<std::vector<Neighbor>>& knNeighbors);
std::pair<double, double> armaFindLSHKNN(const std::vector<std::vector<float>>& database, const std::vector<std::vector<float>>& queries, const int k, std::vector<std::vector<Neighbor>>& knNeighbors);

#endif //UTILS_H
