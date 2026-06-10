#ifndef IVF_ANN
#define IVF_ANN

#include <float.h>
#include "k_means.h"
#include "BST.h"

/*
typedef struct{
    int dim;
    int no_of_data_pts;
    int shape[2]; //shape[0] = no of data pts shape[1] = data pt dim 
    float* data; 
    // access ith data pt as data[i*shape[1]+j] for j = 0 to shape[1]-1
    int* labels; // for assgining data points to clusters
} DATA_FLOAT ;

typedef struct{
    int num_clusters;
    int dim;
    float* centroids;
    int* cluster_sizes;
} CLUSTERS;
*/
typedef struct{
    CLUSTERS* cls;
    int** inverted_lists;
    int* inverted_lists_length;
    DATA_FLOAT * train_data;
    // for training the k_means clustering algorithm
    // usually a portion of search_database;
    DATA_FLOAT * search_database;
} IVF_Index;

typedef struct{
    int idx;
    float dist;
} SEARCH_RESULT;

typedef struct{
    int num_queries;
    int k;
    int* idx;
    float* dist;
} SEARCH_RESULTS;

int LinearSearch(int* array, int array_length, int target);
float Mean(int* array, int num_array);
int compare_search_results(const void *a, const void *b);
IVF_Index* InitIVFIndex(DATA_FLOAT * train_data, int dim, int nlist);
void TrainIVFIndex(IVF_Index* index);
void AddDataBase(IVF_Index* index, DATA_FLOAT * search_database);
void AddDataBaseFuzzy(IVF_Index* index, DATA_FLOAT * search_database);
SEARCH_RESULT* DistancesToClusters(IVF_Index* index, DATA_FLOAT * query_database, int query_id);
SEARCH_RESULT* SearchCluster(IVF_Index* index, DATA_FLOAT * query_database, int cluster_id, int query_id, int k);
SEARCH_RESULT* SearchSingleQuery(IVF_Index* index, DATA_FLOAT * query_database, int query_id, int nprobe, int k);
SEARCH_RESULTS* SearchMultipleQueries(IVF_Index* index, DATA_FLOAT * query_database, int nprobe, int k);
float RecallSingleQuery(int* indices, int* ground_truth, int k);
float RecallMultipleQuery(int* indices, int* ground_truth, int num_queries, int k);
void SearchResultToArray(SEARCH_RESULT* result, float** distances, int** indices,int k);
void printArray(const void *arr, size_t n, size_t elemSize, void (*printElem)(const void *));

#endif