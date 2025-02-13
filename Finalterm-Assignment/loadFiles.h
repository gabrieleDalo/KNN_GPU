#ifndef LOADFILES_H
#define LOADFILES_H
#include "utils.h"
#include <mlpack/core.hpp>

std::vector<std::vector<float>> loadFvecs(const std::string& file);
std::vector<std::vector<int>> loadIvecs(const std::string& file);
void printIntVectors(const std::vector<std::vector<int>>& vectors);
void printFloatVectors(const std::vector<std::vector<float>>& vectors);
arma::mat convertToArmaMatrix(const std::vector<std::vector<float>>& data);
void convertToNeighbors(const arma::Mat<size_t>& lshNeighbors, const arma::mat& lshDistances, std::vector<std::vector<Neighbor>>& knNeighbors);

#endif //LOADFILES_H
