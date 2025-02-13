#include <iostream>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include <vector>
#include <random>

#include "loadFiles.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

using namespace std;

__constant__ float d_queries_constant[128 * 100];   // Massima dimensione della memoria costante, 128 * 128 = 64KB, so che i vettori hanno dimensione 128 quindi posso memorizzarvi fino a 128 query (so che la mia memoria costante ha dimensione 64KB), metto massimo 100 per lasciare un po' di spazio

// Funzione per trasferire i dati dalla CPU alla memoria della GPU
__host__ void transferDataToGPU(const vector<vector<float>>& dataset, const vector<vector<float>>& queries, float** d_dataset, float** d_queries, int ds_size, int num_queries, int nDim){
    cudaError_t err;

    // Flatten dei dati del database e delle query, per memorizzarli meglio nella memoria della gpu
    // Questo tipo di memorizzazione favorisce accessi coalescend (per i threads dei warp)
    vector<float> flat_dataset(ds_size * nDim);
    for(int i = 0; i < ds_size; i++){
        copy(dataset[i].begin(), dataset[i].end(), flat_dataset.begin() + i * nDim);
    }

    vector<float> flat_queries(num_queries * nDim);
    for(int i = 0; i < num_queries; i++){
        copy(queries[i].begin(), queries[i].end(), flat_queries.begin() + i * nDim);
    }

    // Allocazione memoria globale per il dataset
    cudaMalloc(d_dataset, ds_size * nDim * sizeof(float));
    err = cudaMemcpy(*d_dataset, flat_dataset.data(), ds_size * nDim * sizeof(float), cudaMemcpyHostToDevice);

    if(err == cudaSuccess){
        cout << "Dataset loaded into global memory" << endl;
    }else{
        cerr << "Error while loading dataset" << endl;
        cout << cudaGetErrorString(err) << endl;
    }

    // Allocazione memoria delle query, prova a copiare nella memoria costante (è piu' veloce), posso farlo, in generale, se ho poche query
    err = cudaMemcpyToSymbol(d_queries_constant, flat_queries.data(), num_queries * nDim * sizeof(float));  // Copia l'array di queries nell'array che risiede nella memoria costante della GPU (non si usa un puntatore come nel caso della memoria globale)

    if(err == cudaSuccess){     // Se la copia ha successo vuol dire che avevo abbastanza spazio nella memoria costante
        cout << "Queries loaded into constant memory" << endl;
        *d_queries = nullptr;       // Non ho bisogno di memoria globale
    }else{                      // Se la copia fallisce allora li copio nella memoria globale
        cout << "Error while loading queries, passing to global memory..." << endl;
        cudaGetLastError();     // Tolgo l'errore dalla "pila" perchè altrimenti potrebbe dare problemi con thrust

        cudaMalloc(d_queries, num_queries * nDim * sizeof(float));
        err = cudaMemcpy(*d_queries, flat_queries.data(), num_queries * nDim * sizeof(float), cudaMemcpyHostToDevice);

        if(err == cudaSuccess){
            cout << "Queries loaded into global memory" << endl;
        }else{
            cerr << "Error while loading queries" << endl;
            cout << cudaGetErrorString(err) << endl;
        }
    }
}

// Kernel che utilizza la shared memory e la riduzione parallela per calcolare le distanze tra vettori e query
// Le prestazioni in termini di tempo peggiorano all'aumentare del tempo, perchè la shared memory potrebbe non bastare e bloccare temporaneamente l'esecuzione di alcuni blocchi
__global__ void calculateDistancesSharedMemory(float* d_database, float* d_query, Neighbor* d_neighbors, int nDim, int ds_size, int num_queries) {
    extern __shared__ float sharedDist[];  // Memoria condivisa per accumulare le componenti della distanza

    int queryIdx = blockIdx.y;          // Ogni blocco lungo l'asse y gestisce una query
    int vectorIdx = blockIdx.x * blockDim.y + threadIdx.y;  // Ogni vettore è gestito da una colonna di un blocco
    int i = threadIdx.x;        // Ogni thread calcola una componente della distanza
    // Es. il thread (blockIdx.x = 0, blockIdx.y = 1, threadIdx.x = 2, threadIdx.y = 3) calcola la terza componente della distanza tra il quarto vettore del database e la seconda query

    // Indice per la memoria condivisa
    int sharedIdx = threadIdx.y * blockDim.x + threadIdx.x;

    if(vectorIdx < ds_size){
        float diff = 0.0f;

        // Calcola una componente della distanza se il thread è nel range
        if(i < nDim){
            if(d_query == nullptr){     // Controlla dove sono salvate le query
                diff = d_database[vectorIdx * nDim + i] - d_queries_constant[queryIdx * nDim + i];
            }else{
                diff = d_database[vectorIdx * nDim + i] - d_query[queryIdx * nDim + i];
            }
            sharedDist[sharedIdx] = diff * diff;  // Salva il risultato parziale nella memoria condivisa
        }else{
            sharedDist[sharedIdx] = 0.0f;  // I thread fuori dal range nDim scrivono 0
        }

        __syncthreads();  // Sincronizza tutti i thread del blocco per completare il calcolo delle componenti

        // Riduzione parallela nella memoria condivisa
        // Facendo la riduzione con lo stride in questo modo, siamo in grado di: accedere ad elementi vicini nella memoria (coalescend)
        // poichè le somme parziali sono memorizzate tutte vicine, di ridurre il numero di iterazioni e di incrementare il parallelismo (mantiene attivi threads consecutivi)
        // riducendo la divergenza (a differenza del caso con stride incrementale)
        for(int stride = blockDim.x / 2; stride > 0; stride /= 2){
            if(i < stride){
                sharedDist[sharedIdx] += sharedDist[sharedIdx + stride];
            }
            __syncthreads();  // Assicurati che tutti i threads completino ogni passo della riduzione, mettere il sync prima dell'if significherebbe sincronizzare anche thread che non hanno partecipato al calcolo in quell'iterazione
        }

        // Solo il primo thread della riga (threadIdx.x == 0) calcola la distanza totale
        if(threadIdx.x == 0){
            float dist = sqrt(sharedDist[sharedIdx]);  // Calcola la radice quadrata
            d_neighbors[queryIdx * ds_size + vectorIdx].index = vectorIdx;  // Salva l'indice
            d_neighbors[queryIdx * ds_size + vectorIdx].distance = dist;   // Salva la distanza
        }
    }
}

// Kernel che non utilizza la shared memory ma le operazioni atomiche, il fatto che vengano usate molteplici addizioni atomiche può influenzare la precisione dei dati
// A seconda del numero di dati (dataset e query) puo' impiegare meno tempo dell'implementazione con shared memory
__global__ void calculateDistancesAtomic(float* d_database, float* d_query, Neighbor* d_neighbors, int nDim, int ds_size, int num_queries) {
    // Ogni thread calcola la distanza tra un vettore e una query
    int queryIdx = blockIdx.y;          
    int vectorIdx = blockIdx.x * blockDim.y + threadIdx.y;
    int i = threadIdx.x;

    if(vectorIdx < ds_size){
        float diff = 0.0f;

        if(i < nDim){
            if(d_query == nullptr){
                diff = d_database[vectorIdx * nDim + i] - d_queries_constant[queryIdx * nDim + i];
            }else{
                diff = d_database[vectorIdx * nDim + i] - d_query[queryIdx * nDim + i];
            }
            diff = diff * diff;
        }else{
            diff = 0.0f;
        }
        // Calcola la distanza quadrata e accumula nella memoria globale per ogni thread
        atomicAdd(&d_neighbors[queryIdx * ds_size + vectorIdx].distance, diff);  // Accumula nelle distanze quadrate

        __syncthreads();

        if(threadIdx.x == 0){
            d_neighbors[queryIdx * ds_size + vectorIdx].index = vectorIdx;
            d_neighbors[queryIdx * ds_size + vectorIdx].distance = sqrt(d_neighbors[queryIdx * ds_size + vectorIdx].distance);
        }
    }
}

// Kernel in cui ogni thread di un blocco calcola la distanza tra un intero vettore ed una query, non lavora sulla singola componente ma su l'intero vettore
// Quindi non ha bisogno di shared memory per salvare risultati intermedi o di operazioni atomiche, ma non si sfrutta al massimo il parallelismo
__global__ void calculateDistancesVectorThread(float* d_database, float* d_query, Neighbor* d_neighbors, int nDim, int ds_size, int num_queries){
    int queryIdx = blockIdx.y;
    int vectorIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if(vectorIdx < ds_size){
        float dist = 0.0f;

        for(int i = 0; i < nDim; i++){
            float diff = d_database[vectorIdx * nDim + i] - d_query[queryIdx * nDim + i];
            if(d_query == nullptr){
                diff = d_database[vectorIdx * nDim + i] - d_queries_constant[queryIdx * nDim + i];
            }else{
                diff = d_database[vectorIdx * nDim + i] - d_query[queryIdx * nDim + i];
            }
            dist += diff * diff;
        }

        dist = sqrt(dist);

        d_neighbors[queryIdx * ds_size + vectorIdx].index = vectorIdx;
        d_neighbors[queryIdx * ds_size + vectorIdx].distance = dist;
    }
}

int main(){
    double startTimeDataTransfer, endTimeDataTransfer;
    double startTime, exeTime, exeTimeSequential, exeTimeParallel;
    double speedupDistances, speedupKNN;
    pair<double, double> times;     // Tempi di calcolo delle distanze e sort

    // Carica database e query
    string dbFile, queryFile, testsFile;
    int dbSize, nQueries, vectorSize;
    int k = 100;                    // Numero di vicini più prossimi da cercare
    float testAccuracy = 0;
    bool correctInput = false, doTests = false;
    vector<vector<float>> database, queries;
    vector<vector<int>> tests;                  // Contiene, per ogni query, gli indici dei KNN
    vector<vector<float>> result, resultq;
    int n = 300000;             // Il numero di vettori del dataset che vuoi estrarre, se richiesto
    int nq = 5000;                  // Il numero di vettori di query che vuoi estrarre

    // Faccio selezionare all'utente il dataset da usare
    int input;
    cout << "Datasets:" << endl;
    cout << " 1. Small" << endl;
    cout << " 2. Medium" << endl;
    cout << " 3. Big" << endl;
    cout << "Choose dataset to use: ";
    while(!correctInput){
        cin >> input;

        if(cin.fail() || input < 1 || input > 3){
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "Invalid input. Reinsert: ";
        }else{
            correctInput = true;
        }
    }

    // Carico il dataset scelto dall'utente
    switch(input){
        case 1:
            dbFile = "siftsmall/siftsmall_base.fvecs";
            queryFile = "siftsmall/siftsmall_query.fvecs";
            testsFile = "siftsmall/siftsmall_groundtruth.ivecs";

            database = loadFvecs(dbFile);     // Carica i vettori dai files
            queries = loadFvecs(queryFile);

            doTests = true;
            break;
        case 2:
            dbFile = "sift/sift_base.fvecs";
            queryFile = "sift/sift_query.fvecs";

            database = loadFvecs(dbFile);
            queries = loadFvecs(queryFile);

            // Riduco i dataset, non posso fare i tests poichè non ho groundtruth
            std::copy(database.begin(), database.begin() + std::min(n, (int)database.size()), std::back_inserter(result));
            database.clear();
            database = result;

            std::copy(queries.begin(), queries.begin() + std::min(nq, (int)queries.size()), std::back_inserter(resultq));
            queries.clear();
            queries = resultq;

            doTests = false;
            break;
        case 3:
            dbFile = "sift/sift_base.fvecs";
            queryFile = "sift/sift_query.fvecs";
            testsFile = "sift/sift_groundtruth.ivecs";

            database = loadFvecs(dbFile);
            queries = loadFvecs(queryFile);

            doTests = true;
            break;
        default:
            cerr << "Error" << endl;
            break;
    }

    dbSize = database.size();
    nQueries = queries.size();
    vectorSize = database[0].size();

    if(dbSize == 0 || nQueries == 0){
        cerr << "Error while loading data" << endl;
        return 1;
    }
    if(doTests){            // Carico i groundtruth, se possibile
        tests = loadIvecs(testsFile);

        if(tests.size() == 0){
            cerr << "Error while loading data" << endl;
            return 1;
        }
    }

    correctInput = false;
    // Faccio selezionare all'utente il tipo di metodo da utilizzare per il KNN
    cout << "Settings:" << endl;
    cout << " 1. Shared memory" << endl;
    cout << " 2. Atomic" << endl;
    cout << " 3. One vector per thread" << endl;
    cout << "Choose CUDA parallel implementation: ";
    while(!correctInput){
        cin >> input;

        if(cin.fail() || input < 1 || input > 3){
            cin.clear();
            cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            cout << "Invalid input. Reinsert: ";
        }else{
            correctInput = true;
        }
    }
    
    vector<vector<Neighbor>> knNeighborsParallel(nQueries);     // Struttura che conterrà, per ogni query, i K vettori del dataset piu' vicini
    vector<vector<Neighbor>> knNeighborsSequential(nQueries);

    //printFloatVectors(database);
    //printFloatVectors(queries);
    //printIntVectors(tests);
    cout << "" << endl;
    cout << "Dataset size: " << dbSize << " vectors with " << vectorSize << " dimensions" << endl;
    cout << "Number of queries: " << nQueries << endl;
    cout << "Number of test vectors: " << tests.size() << endl;
    cout << "Number of neighbors to search (K) =  " << k << endl;
    cout << "" << endl;

    startTime = omp_get_wtime();
    
    // Parametri per CUDA
    cudaError_t err;
    float* gpu_database_ptr = nullptr;      // Puntatori alla memoria globale della GPU per dataset e query
    float* gpu_queries_ptr = nullptr;
    int deviceCount;                    // Parametri per ottenere le specifiche della GPU principale installata sul pc
    cudaGetDeviceCount(&deviceCount);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    cout << "--------------- Exact KNN (parallel) ---------------" << endl;
    cout << "" << endl;

    // Trasferimento dei dati alla GPU
    cout << "Loading data into the GPU..." << endl;
    startTimeDataTransfer = omp_get_wtime();
    transferDataToGPU(database, queries, &gpu_database_ptr, &gpu_queries_ptr, dbSize, nQueries, vectorSize);
    endTimeDataTransfer = omp_get_wtime() - startTimeDataTransfer;
    // Non serve sincronizzare esplicitamente poichè le cudaMemcpy sono sincrone di default
    cout << "" << endl;

    // Calcola il numero massimo di query che possono essere elaborate in un batch, a seconda della memoria disponibile
    size_t memFree, memTotal;
    cudaMemGetInfo(&memFree, &memTotal);

    size_t perQueryMemory = dbSize * sizeof(Neighbor);      // Memoria necessaria per i risultati di una query
    size_t availableMemory = memFree * 0.95;             // Usa al max il 95% della memoria libera
    int batchSize;                                          // Numero massimo di query per batch

    if(nQueries * dbSize * sizeof(Neighbor) < availableMemory){     // Controllo se c'è memoria libera disponibile per tutte le query
        batchSize = nQueries;
    }else{                                                      // Se non c'è, eseguo le query in batch
        batchSize = availableMemory / perQueryMemory;
    }

    if(batchSize == 0){
        cerr << "Not enough memory to process even one query!" << endl;
        return 1;
    }

    cout << "Num. batch: " << ceil(static_cast<float>(nQueries)/ static_cast<float>(batchSize)) << ", batch size: " << batchSize << " queries per batch" << endl;

    // Configura la griglia e i blocchi
    double totalTimeComputeDist = 0.0, totalTimeKNN = 0.0;
    int vectorDim = vectorSize;                             // Numero di componenti di un vettore
    int nVectorsPerBlock = prop.maxThreadsPerBlock;        // Numero di vettori per blocco

    size_t sharedMemSize;    // Dimensione della memoria condivisa (shared memory) di un blocco
    size_t memoryPerVector;        
    int maxVectorsByMemory;      
    int maxVectorsByThreads;

    Neighbor* d_neighbors;           // Salvo sulla memoria globale della GPU, per ogni query della batch in esecuzione, indice e distanza di tutti i vettori da quella query
    cudaMalloc(&d_neighbors, batchSize * dbSize * sizeof(Neighbor));

    // Stampo lo stato della memoria globale della GPU
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("GPU available memory: %.2f MB, Total: %.2f MB, In use: %.2f MB\n", (free_mem / (1024.0f * 1024.0f)), (total_mem / (1024.0f * 1024.0f)), ((total_mem / (1024.0f * 1024.0f)) - (free_mem / (1024.0f * 1024.0f))));
    cout << "" << endl;

    dim3 blockDim;
    dim3 gridDim;

    // Imposto dimensioni diverse per griglia e blocco a seconda della modalità di parallelizzazione con CUDA
    switch(input){
        case 1:
            memoryPerVector = vectorDim * sizeof(float);         // Memoria richiesta per un singolo vettore
            maxVectorsByMemory = prop.sharedMemPerBlock / memoryPerVector;       // Calcola quanti vettori possono essere memorizzati in shared memory
            maxVectorsByThreads = prop.maxThreadsPerBlock / vectorDim;     // Calcola il numero massimo di vettori dato il limite dei thread
            nVectorsPerBlock = min(maxVectorsByMemory, maxVectorsByThreads);    // Determina il numero massimo di vettori per blocco considerando entrambi i limiti

            blockDim.x = vectorDim;              // Organizza sull'asse x del blocco le componenti dei vettori, e sull'asse y i vettori
            blockDim.y = nVectorsPerBlock;
            
            gridDim.x = ceil(static_cast<float>((dbSize + nVectorsPerBlock - 1)) / static_cast<float>(nVectorsPerBlock));     // Organizza sull'asse x i blocchi e sull'asse y le query
            gridDim.y = batchSize;

            sharedMemSize = vectorDim * nVectorsPerBlock * sizeof(float);    // Dimensione della memoria condivisa (shared memory) di un blocco

            break;
        case 2:
            // Come caso precedente, ma le dimensioni non dipendono dalla quantità di shared memory disponibile
            blockDim.x = vectorDim;
            blockDim.y = nVectorsPerBlock/vectorDim;
            
            gridDim.x = ceil(static_cast<float>((dbSize + (nVectorsPerBlock/vectorDim) - 1)) / static_cast<float>((nVectorsPerBlock/vectorDim)));     // Organizza sull'asse x i blocchi e sull'asse y le query
            gridDim.y = batchSize;

            break;
        case 3:
            blockDim.x = 1;     // Ogni thread calcola la distanza per un singolo vettore (non lavora sulla singola componente)
            blockDim.y = nVectorsPerBlock;

            gridDim.x = ceil(static_cast<float>((dbSize + nVectorsPerBlock - 1)) / static_cast<float>(nVectorsPerBlock));     // Organizza sull'asse x i blocchi e sull'asse y le query
            gridDim.y = batchSize;
            break;
        default:
            cerr << "Error" << endl;
            break;
    }

    // Avvio il kernel per effettuare la KNN
    cout << "Launching kernels..." << endl;

    // Processa i batch
    for(int batchStart = 0; batchStart < nQueries; batchStart += batchSize){
        int currentBatchSize = min(batchSize, nQueries - batchStart);       // Lo faccio, in caso non tutti i batch abbiano la stessa dimensione (es. l'ultimo è piu' piccolo)
        double time;
        gridDim.y = currentBatchSize;
        
        cout << "" << endl;
        cout << "Processing batch " << (batchStart / batchSize + 1) << " (" << currentBatchSize << " queries)..." << endl;
        exeTime = omp_get_wtime();

        // Eseguo il kernel
        switch(input){
            case 1:
                calculateDistancesSharedMemory<< <gridDim, blockDim, sharedMemSize >> >(gpu_database_ptr, gpu_queries_ptr + batchStart * vectorDim, d_neighbors, vectorSize, dbSize, currentBatchSize);
                break;
            case 2:
                calculateDistancesAtomic<< <gridDim, blockDim >> >(gpu_database_ptr, gpu_queries_ptr + batchStart * vectorDim, d_neighbors, vectorSize, dbSize, currentBatchSize);
                break;
            case 3:
                calculateDistancesVectorThread<< <gridDim, blockDim >> >(gpu_database_ptr, gpu_queries_ptr + batchStart * vectorDim, d_neighbors, vectorSize, dbSize, currentBatchSize);
                break;
            default:
                cerr << "Error" << endl;
                break;
        }
        cudaDeviceSynchronize();        // Assicura che il kernel sia terminato

        cout << "Time required to compute distances: ";
        time = omp_get_wtime() - exeTime;
        printTime(time);
        totalTimeComputeDist += time;

        cout << "Finding K nearest neighbors..." << endl;
        exeTime = omp_get_wtime();

        // Ordina i risultati e copia i K vicini per ogni query
        for(int q = 0; q < currentBatchSize; q++){
            int queryIdx = batchStart + q;
            thrust::device_ptr<Neighbor> neighbors(d_neighbors + q * dbSize);

            thrust::sort(neighbors, neighbors + dbSize, [] __device__(const Neighbor & a, const Neighbor & b){
                if(a.distance == b.distance){
                    return a.index < b.index;  // Ordinamento per indice in caso di parità nelle distanze
                }
                return a.distance < b.distance;
            });

            vector<Neighbor> query_neighbors(k);
            startTimeDataTransfer = omp_get_wtime();
            cudaMemcpy(query_neighbors.data(), d_neighbors + q * dbSize, k * sizeof(Neighbor), cudaMemcpyDeviceToHost);
            knNeighborsParallel[queryIdx] = move(query_neighbors);
            // Basta spostare i dati, non serve cancellarli dalla GPU perchè tanto verranno sovrascritti alla prossima batch
            endTimeDataTransfer += omp_get_wtime() - startTimeDataTransfer;
        }

        cout << "Time required to find the KNN: ";
        time = omp_get_wtime() - exeTime;
        printTime(time);
        totalTimeKNN += time;
    }
    
    cout << "" << endl;
    cout << "Kernels execution completed" << endl;
    cout << "" << endl;

    // Pulisco la memoria della GPU
    cout << "Cleaning GPU memory..." << endl;

    err = cudaFree(gpu_database_ptr);
    if(err != cudaSuccess){
        cerr << "Error while deleating GPU memory for dataset" << endl;
        cout << cudaGetErrorString(err) << endl;
    }
    if(gpu_queries_ptr){        // Se le query sono salvate nella memoria costante non serve cancellarle esplicitamente
        err = cudaFree(gpu_queries_ptr);
        if(err != cudaSuccess){
            cerr << "Error while deleating GPU memory for queries" << endl;
            cout << cudaGetErrorString(err) << endl;
        }
    }
    err = cudaFree(d_neighbors);
    if(err != cudaSuccess){
        cerr << "Error while deleating GPU memory for neighbors" << endl;
        cout << cudaGetErrorString(err) << endl;
    }

    exeTimeParallel = omp_get_wtime() - startTime;
    /*
    // Stampo i risultati
    cout << "" << endl;
    for(size_t i = 0; i < knNeighborsParallel.size(); i++) {  // Itera su ogni query
        cout << "KNN query " << (i + 1) << ": " << endl;   // Stampa l'indice della query

        // Itera sui K vicini per la query i-esima
        for(size_t j = 0; j < knNeighborsParallel[i].size(); j++){
            cout << "  Vector index: " << knNeighborsParallel[i][j].index << ", distance: " << knNeighborsParallel[i][j].distance << endl;
        }

        cout << "" << endl;  // Linea vuota tra le query
    }
    */
    // Se possibile, confronto i risultati con i groundtruth
    if(doTests){
        cout << "" << endl;
        cout << "Comparing with groundtruth..." << endl;
        testAccuracy = compareGroundtruth(dbSize, tests, k, knNeighborsParallel);
        cout << "Test accuracy exact KNN (parallel): " << testAccuracy << "%" << endl;
    }

    speedupDistances = totalTimeComputeDist;
    speedupKNN = totalTimeKNN;

    cout << "" << endl;
    cout << "Time required to compute distances: ";
    printTime(totalTimeComputeDist);
    cout << "Time required to find the KNN: ";
    printTime(totalTimeKNN);
    cout << "Time required to transfer data: ";
    printTime(endTimeDataTransfer);
    cout << "Exact KNN execution time (parallel): ";
    printTime(exeTimeParallel);
    
    // KNN versione sequenziale
    cout << "" << endl;
    cout << "--------------- Exact KNN (sequential) ---------------" << endl;
    cout << "Searching..." << endl;
    cout << "" << endl;

    startTime = omp_get_wtime();
    times = findKNN(database, queries, k, knNeighborsSequential);
    exeTimeSequential = omp_get_wtime() - startTime;
    /*
    // Stampa tutti i vicini trovati
    for(size_t i = 0; i < nQueries; i++){  // Itera su ogni query
        cout << "KNN query " << (i + 1) << ": " << endl;   // Stampa l'indice della query

        // Itera sui K vicini per la query i-esima
        for(size_t j = 0; j < knNeighborsSequential[i].size(); j++){
            cout << "  Vector index: " << knNeighborsSequential[i][j].index << ", distance: " << knNeighborsSequential[i][j].distance << endl;
        }

        cout << endl;  // Linea vuota tra le query
    }
    */
    if(doTests){
        cout << "Comparing with groundtruth..." << endl;
        testAccuracy = compareGroundtruth(dbSize, tests, k, knNeighborsSequential);      // Confronto i risultati con i groundtruth
        cout << "Test accuracy exact KNN (sequential): " << testAccuracy << "%" << endl;
        cout << "" << endl;
    }

    speedupDistances = times.first / speedupDistances;
    speedupKNN = times.second / speedupKNN;
    
    cout << "Time required to compute distances: ";
    printTime(times.first);

    cout << "Time required to find the KNN: ";
    printTime(times.second);

    cout << "Exact KNN execution time (sequential): ";
    printTime(exeTimeSequential);

    // Se non ho i groundtruth confronto l'accuratezza tra le 2 versioni
    if(!doTests && !knNeighborsSequential.empty() && !knNeighborsParallel.empty()){
        float precision = calculatePrecision(knNeighborsSequential, knNeighborsParallel);

        if(precision >= 0.0f){
            cout << "Precision beetween sequential and parallel version: " << precision << "%" << endl;
        }else{
            cout << "An error occurred while calculating precision." << endl;
        }
    }
    /*
    // KNN versione sequenziale con LSH
    cout << "" << endl;
    cout << "--------------- KNN LSH (sequential) ---------------" << endl;

    // Svuoto il vettore se devo riusarlo
    for(auto& neighbors : knNeighborsSequential){
        neighbors.clear();
    }
    startTime = omp_get_wtime();
    times = armaFindLSHKNN(database, queries, tests, k, knNeighborsSequential);
    exeTime = omp_get_wtime() - startTime;

    if(doTests){
        cout << "Comparing with groundtruth..." << endl;
        testAccuracy = compareGroundtruth(dbSize, tests, k, knNeighborsSequential);      // Confronto i risultati con i groundtruth
        cout << "Test accuracy KNN LSH (sequential): " << testAccuracy << "%" << endl;
        cout << "" << endl;
    }
    cout << "Time required to create LSH: ";
    printTime(times.second);

    cout << "Time required to find the KNN: ";
    printTime(times.first);
    cout << "" << endl;

    cout << "LSH KNN execution time (sequential): ";
    printTime(exeTime);
    */
    cout << "" << endl;
    cout << "Exact KNN execution time (sequential): ";
    printTime(exeTimeSequential);
    cout << "Exact KNN execution time (parallel): ";
    printTime(exeTimeParallel);
    cout << "" << endl;

    cout << "Compute distances Speedup: " << speedupDistances << endl;
    cout << "Find KNN Speedup: " << speedupKNN << endl;
    cout << "Total Speedup: " << exeTimeSequential / exeTimeParallel << endl;

    return 0;
}

