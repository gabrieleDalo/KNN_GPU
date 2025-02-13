#include <vector>
#include <cmath>
#include <omp.h>
#include <algorithm>
#include "utils.h"

#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/lsh/lsh_search.hpp>
#include "loadFiles.h"

using namespace std;

// Dato il tempo di esecuzione in secondi, stampa i secondi, i minuti ed i millisecondi impiegati
void printTime(double time) {
    cout << static_cast<int>(time * 1000) / 60000 << " minutes, "
    << (static_cast<int>(time * 1000) / 1000) % 60 << " seconds, and "
    << static_cast<int>(time * 1000) % 1000 << " milliseconds" << endl;
}

// Calcola la distanza euclidea tra 2 vettori
float euclideanDistance(const vector<float>& a, const vector<float>& b){
    float sum = 0.0f;
    for(vector<float>::size_type i = 0; i < a.size(); i++){
        float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Funzione per calcolare il grado di precisione (differenza tra 2 vettori)
float calculatePrecision(const vector<vector<Neighbor>>& knNeighbors1, const vector<vector<Neighbor>>& knNeighbors2){
    float precision = 1;

    // Verifica che i due vettori abbiano la stessa dimensione
    if(knNeighbors1.size() != knNeighbors2.size()){
        cerr << "Error: The sizes of the two vectors do not match!" << endl;
        return 1;
    }

    int totalComparisons = 0;
    int correctMatches = 0;

    // Itera su ogni query
    for(size_t i = 0; i < knNeighbors1.size(); i++){
        // Verifica che i sottovettori abbiano la stessa dimensione
        if(knNeighbors1[i].size() != knNeighbors2[i].size()){
            cerr << "Error: The sizes of sub-vectors at index " << i << " do not match!" << endl;
            return -1.0f;
        }

        // Itera sugli elementi dei sottovettori
        for(size_t j = 0; j < knNeighbors1[i].size(); j++){
            const Neighbor& n1 = knNeighbors1[i][j];
            const Neighbor& n2 = knNeighbors2[i][j];

            // Confronta l'indice e la distanza con tolleranza
            if(n1.index == n2.index && abs(n1.distance - n2.distance) <= precision){
                correctMatches++; // Match corretto
            }
            totalComparisons++; // Incrementa il totale dei confronti
        }
    }

    // Calcola il grado di precisione in percentuale
    return (static_cast<float>(correctMatches) / totalComparisons) * 100.0f;
}

// Controlla se i dati passati sono uguali a quelli di test
float compareGroundtruth(const int dbSize, const vector<vector<int>>& tests, const int k, vector<vector<Neighbor>>& exactNeighbors){
    int c_errors = 0;
    bool error = false;
    float accuracy;

    for(vector<float>::size_type i = 0; i < tests.size(); i++){
        for(vector<float>::size_type y = 0; y < exactNeighbors[i].size(); y++) {
            if(exactNeighbors[i][y].index != tests[i][y]){        // Controllo che i risultati siano uguali a quelli dei tests
                int p = static_cast<int>(y + 1);
                error = true;
                while(p > 0 && p < exactNeighbors.size() && exactNeighbors[i][p].distance == exactNeighbors[i][y].distance){      // Se non lo sono, controllo che non sia un problema di ordinamento in caso di distanze uguali
                    if(exactNeighbors[i][p].index == tests[i][y]){
                        error = false;
                        break;
                    }
                    p++;
                }
                if(error){
                    p = static_cast<int>(y - 1);
                    while(p > 0 && p < exactNeighbors.size() && exactNeighbors[i][p].distance == exactNeighbors[i][y].distance){      // Se non lo sono, controllo che non sia un problema di ordinamento in caso di distanze uguali
                        if(exactNeighbors[i][p].index == tests[i][y]){
                            error = false;
                            break;
                        }
                        p--;
                    }
                }
                if(error){
                    //cout << "Query " << (i + 1) << endl;
                    //cout << "  Results different from groundtruth, Vector " << (y + 1) << " index: " << exactNeighbors[i][y].index << " (distance: " << exactNeighbors[i][y].distance << "), different from " << tests[i][y] << endl;
                    c_errors++;
                }
            }
        }
    }

    accuracy = 100 - (static_cast<float>(c_errors) / static_cast<float>(dbSize)) * 100;

    return accuracy;
}

// KNN classico, metodo per trovare i k vicini più vicini ad un dato vettore di query
// Usando il sorting standard di std
pair<double, double> findKNN(const vector<vector<float>>& database, const vector<vector<float>>& queries, const int k, vector<vector<Neighbor>>& knNeighbors){
    double startExeTime, startSortTime, totalSortTime = 0.0;
    vector<Neighbor> exactNeighbors;

    startExeTime = omp_get_wtime();

    for(vector<float>::size_type i = 0; i < queries.size(); i++){
        if((i+1)%100 == 0)
            cout << "Query " << (i + 1) << ":\n";
        // Calcola la distanza tra ogni vettore del dataset e la query
        for(vector<float>::size_type j = 0; j < database.size(); j++){
            float dist = euclideanDistance(database[j], queries[i]);
            exactNeighbors.push_back({static_cast<int>(j), dist});
        }

        startSortTime = omp_get_wtime();
        // Ordina per distanza crescente, e in caso di parità per indice crescente
        sort(exactNeighbors.begin(), exactNeighbors.end(),
            [](const Neighbor& a, const Neighbor& b) {
                if(a.distance == b.distance){
                    return a.index < b.index;  // Ordinamento per indice in caso di parità nelle distanze
                }
                return a.distance < b.distance;  // Ordinamento per distanza
            });

        exactNeighbors.resize(k);        // Restituisci solo i primi k vicini

        knNeighbors[i] = exactNeighbors;
        exactNeighbors.clear();

        totalSortTime += omp_get_wtime() - startSortTime;
    }

    return make_pair((omp_get_wtime() - startExeTime) - totalSortTime, totalSortTime);
}

// Comparatore per il min-heap (ordinato per distanza crescente e indice decrescente)
// Si usa perchè come standard la priority queue è organizzata come max-heap
struct CompareNeighbor{
    bool operator()(const Neighbor& a, const Neighbor& b){
        if(a.distance == b.distance){
            return a.index < b.index;
        }
        return a.distance < b.distance;
    }
};

// Trova i k vicini più vicini per ogni query, usando un min-heap (il valore di ogni nodo è minore o uguale a quello dei suoi (2) figli)
// Quando si inserisce un nuovo elemento, la priority_queue lo aggiunge alla fine dell'heap e poi lo riordina risalendo l'albero (processo detto heapify-up)
// La rimozione estrae sempre l'elemento con priorità più alta (nel nostro caso, il più piccolo perché usiamo un min-heap). Il processo avviene così: 
// sostituiamo la radice con l'ultimo elemento, riorganizziamo l'heap (heapify-down): confrontiamo il nuovo nodo con i suoi figli e se è maggiore di un figlio, scambiamo e ripetiamo il processo fino a ristabilire la proprietà dell'heap.
pair<double, double> findKNN_minHeap(const vector<vector<float>>& database, const vector<vector<float>>& queries, const int k, vector<vector<Neighbor>>& knNeighbors){ 
    double startExeTime, startSortTime, totalSortTime = 0.0;
    startExeTime = omp_get_wtime();

    for(size_t i = 0; i < queries.size(); i++){
        // priority_queue<T, Container, Compare> queueName;, T = è il tipo degli elementi nella coda, Container è il contenitore sottostante (di default è vector<T>), Compare è il functor o comparatore usato per definire l'ordine degli elementi
        // Internamente è implementata come un heap binario (albero binario completo)
        priority_queue<Neighbor, vector<Neighbor>, CompareNeighbor> minHeap;

        // Calcola la distanza per ogni punto nel database
        for(size_t j = 0; j < database.size(); j++){
            float dist = euclideanDistance(database[j], queries[i]);
            minHeap.push({ static_cast<int>(j), dist });
            // Vengono inseriti in ordine crescente di distanza, quindi l'ultimo sarà quello piu' lontano

            // Mantieni solo i primi k più vicini, quindi se l'heap è "pieno" toglie l'ultimo elemento che sfora
            if(minHeap.size() > k){
                minHeap.pop();
            }
        }

        startSortTime = omp_get_wtime();

        // Estrae i k vicini e li ordina, crea il vero vettore di KNN estraendoli dal min-heap
        vector<Neighbor> neighbors;
        while(!minHeap.empty()){
            neighbors.push_back(minHeap.top());
            minHeap.pop();
        }
        reverse(neighbors.begin(), neighbors.end()); // Ordine crescente

        knNeighbors[i] = neighbors;
        totalSortTime += omp_get_wtime() - startSortTime;
    }

    return make_pair((omp_get_wtime() - startExeTime) - totalSortTime, totalSortTime);
}

// ANN retrival, metodo per trovare i k vicini più vicini ad un dato vettore di query usando l'approssimazione
pair<double, double> armaFindLSHKNN(const vector<vector<float>>& database, const vector<vector<float>>& queries, const int k, vector<vector<Neighbor>>& knNeighbors) {
    // Converte i dati in matrici Armadillo
    // Le matrici hanno dimensione data da (dimensione vettore)x(n°vettori), es. 128 x 1000000
    arma::mat dataset = convertToArmaMatrix(database);
    arma::mat querie_s = convertToArmaMatrix(queries);
    double startExeTime, startLSHTime, totalLSHTime = 0.0;;

    // Parametri di LSH
    size_t numProjections = 20;     // Aumentando le proiezioni i bucket diventano più selettivi (contengono meno punti) e riduce i falsi positivi ma aumenta i falsi negativi.
    size_t numTables = 1500;        // Aumentando il numero di tabelle migliora la probabilità di trovare i veri vicini (riduce il rischio di punti persi) ma aumenta il costo computazionale e di memoria.

    startExeTime = omp_get_wtime();

    // Istanzia il modello LSH
    cout << "Creating LSH..." << endl;
    startLSHTime = omp_get_wtime();
    mlpack::LSHSearch<> lsh(dataset, numProjections, numTables);
    totalLSHTime = omp_get_wtime() - startLSHTime;

    // Matrici per memorizzare risultati e distanze
    // Le matrici hanno dimensione data da (n° vicini da cercare)x(n° query), es. 100 x 10000
    // Quindi ogni colonna rappresenta i k nearest neighbor per una query
    arma::Mat<size_t> lshNeighbors;  // Indici dei vicini
    arma::mat lshDistances;          // Distanze dei vicini

    // Esegui la ricerca dei vicini più prossimi per ogni query
    cout << "Searching..." << endl;
    cout << "" << endl;
    lsh.Search(querie_s, k, lshNeighbors, lshDistances);

    convertToNeighbors(lshNeighbors, lshDistances, knNeighbors);    // Converto le matrici armadillo dei risultati

    return make_pair((omp_get_wtime() - startExeTime) - totalLSHTime, totalLSHTime);
}