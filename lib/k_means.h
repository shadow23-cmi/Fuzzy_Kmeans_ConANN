#ifndef KMEANS
#define KMEANS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>

typedef struct{
    int dim;
    int no_of_data_pts;
    int shape[2]; //shape[0] = no of data pts shape[1] = data pt dim 
    float* data; 
    // access ith data pt as data[i*shape[1]+j] for j = 0 to shape[1]-1
    int* labels; // for assgining data points to clusters
} DATA_FLOAT ;

typedef struct{
    int dim;
    int no_of_data_pts;
    int shape[2]; //shape[0] = no of data pts shape[1] = data pt dim 
    int* data;
    // access ith data pt as data[i*shape[1]+j] for j = 0 to shape[1]-1
    int* labels; // for assgining data points to clusters
} DATA_INT ;

typedef struct{
    int num_clusters;
    int dim;
    float* centroids;
    int* cluster_sizes;
} CLUSTERS;

float l2_distance(float* a, float* b, int dim);
float RandomFfloat(float min, float max);
float* read_fvecs(const char* filename, int* num_vectors, int* dimension);
int* read_ivecs(const char* filename, int* num_vectors, int* dimension);
DATA_FLOAT * populate_flat_indexed_data_float(DATA_FLOAT * dataset, float* data, int dim, int no_of_data_pts);
DATA_INT * populate_flat_indexed_data_int(DATA_INT * dataset, int* data, int dim, int no_of_data_pts);
void update_centroids(DATA_FLOAT * dataset, CLUSTERS* cls);
void initialize_centroids(DATA_FLOAT * dataset, CLUSTERS* cls, int num_clusters);
int assign_clusters(DATA_FLOAT * dataset, CLUSTERS* cls);
int* assign_fuzzy_clusters(DATA_FLOAT  *dataset, CLUSTERS *cls);
void train_kmeans(DATA_FLOAT  *train_dataset, CLUSTERS *cls);
float compute_sse(DATA_FLOAT  *dataset, CLUSTERS *cls);

#endif