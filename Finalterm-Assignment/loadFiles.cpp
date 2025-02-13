#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include "loadFiles.h"

#include "utils.h"
#include <mlpack/methods/lsh/lsh_search.hpp>

using namespace std;

// Funzione per leggere un file .fvecs
// Legge un file binario nel formato .fvecs, che memorizza vettori di numeri floating-point con una dimensione prefissata per ciascun vettore
vector<vector<float>> loadFvecs(const string& file){
    std::ifstream input(file, std::ios::binary);    // Legge il file in modalità binaria, necessaria per interpretare raw data
    bool error = false;

    if(!input){     // Controlla se il file esiste ed è accessibile
        cout << "Error while opening file: " << file << endl;
        error = true;
    }

    vector<vector<float>> data;
    while(!error && !input.eof()){    // Legge fino alla fine del file
        int dim;                            // Legge la dimensione del vettore
        input.read(reinterpret_cast<char*>(&dim), sizeof(int));     // reinterpret_cast<char*>, converte il puntatore int* in char* per la lettura binaria, sizeof(int), specifica la quantità di byte da leggere
        if(input.eof()) break;     // Per evitare lettura extra a fine file

        // Legge il contenuto del vettore (preceduto dalla sua dimensione)
        vector<float> vec;
        vec.reserve(dim);
        vec.resize(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        data.push_back(std::move(vec));     // std::move(vec), trasferisce il contenuto di vec a data senza copiarlo, migliorando l'efficienza
    }
    input.close();

    return data;
}

// Funzione per leggere un file .ivecs
// E' come prima, ma le componenti dei vettori sono int e non float
vector<vector<int>> loadIvecs(const string& file){
    std::ifstream input(file, std::ios::binary);
    bool error = false;

    if(!input){
        cout << "Error while opening file: " << file << endl;
        error = true;
    }

    vector<vector<int>> data;

    while(!error && !input.eof()){
        int dim;
        input.read(reinterpret_cast<char*>(&dim), sizeof(int));
        if (input.eof()) break;

        vector<int> vec;
        vec.reserve(dim);
        vec.resize(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        data.push_back(std::move(vec));
    }
    input.close();

    return data;
}

// Funzione per stampare i vettori e le loro componenti
void printIntVectors(const vector<vector<int>>& vectors){
    for(vector<int>::size_type i = 0; i < vectors.size(); i++){
        cout << "Size: " << vectors[i].size() << ", Vector " << (i+1) << ": ";
        for(int val : vectors[i]){
            cout << val << " ";
        }
        cout << "\n";
    }
}

void printFloatVectors(const vector<vector<float>>& vectors){
    for(vector<float>::size_type i = 0; i < vectors.size(); i++){
        cout << "Size: " << vectors[i].size() << ", Vector " << (i+1) << ": ";
        for(float val : vectors[i]){
            cout << val << " ";
        }
        cout << "\n";
    }
}

// Funzione per convertire un std::vector<std::vector<float>> in arma::mat (matrice Armadillo necessaria per mlpack)
arma::mat convertToArmaMatrix(const vector<vector<float>>& data){
    if(data.empty()){
        throw std::runtime_error("Il dataset è vuoto!");
    }

    // Numero di righe e colonne
    size_t numRows = data[0].size();  // Dimensione dei vettori
    size_t numCols = data.size();    // Numero di vettori

    arma::mat matrix(numRows, numCols);

    // Copia i dati nella matrice
    for(vector<vector<float>>::size_type col = 0; col < numCols; col++){
        if(data[col].size() != numRows){
            throw std::runtime_error("I vettori nel dataset non hanno tutti la stessa dimensione!");
        }
        for(vector<vector<float>>::size_type row = 0; row < numRows; row++){
            matrix(row, col) = data[col][row];
        }
    }

    return matrix;
}

// Funzione per convertire arma::Mat<size_t> e arma::mat in std::vector<std::vector<Neighbor>>
void convertToNeighbors(const arma::Mat<size_t>& lshNeighbors, const arma::mat& lshDistances, vector<vector<Neighbor>>& knNeighbors) {

    // Verifica che le dimensioni delle matrici siano coerenti
    if(lshNeighbors.n_rows != lshDistances.n_rows || lshNeighbors.n_cols != lshDistances.n_cols){
        cout << "Le dimensioni di lshNeighbors e lshDistances non corrispondono!" << endl;
    }

    // Numero di query (colonne) e numero di neighbor (righe)
    size_t numQueries = lshNeighbors.n_cols;
    size_t numNeighbors = lshNeighbors.n_rows;

    // Conversione dei dati
    for(size_t col = 0; col < numQueries; ++col){
        vector<Neighbor> neighborsForQuery;
        for(size_t row = 0; row < numNeighbors; ++row){
            Neighbor neighbor;
            neighbor.index = static_cast<int>(lshNeighbors(row, col)); // Indice del vicino
            neighbor.distance = static_cast<float>(lshDistances(row, col)); // Distanza del vicino
            neighborsForQuery.push_back(neighbor);
        }
        knNeighbors[col] = move(neighborsForQuery);
    }
}