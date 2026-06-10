#ifndef ConANN
#define ConANN

#include <math.h>
#include "IVF_ANN.h"
#include "BST.h"

#define MAX(a,b) (((a)>(b))?(a):(b))
/*
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
*/

float l2_norm(float* a, int dim);
float GlobalMaxDistanceL2(DATA_FLOAT* dataset);
float* Calibrartion(IVF_Index *index, DATA_FLOAT *calibration_dataset, DATA_INT* calibration_groundtruth,
                     float* D_min,float* D_max,int k, float* alphas, int num_alphas,float gamma, int c_reg);
SEARCH_RESULT *SearchSingleQueryAdaptive(IVF_Index *index, DATA_FLOAT *query_database,
                                         int query_id, float D_min,float D_max,float threshold, int k, float gamma, int c_reg,int* nprobe);

SEARCH_RESULTS *SearchMultipleQueriesAdaptive(IVF_Index *index, DATA_FLOAT *query_database,
                                                 float D_min,float D_max,float threshold, int k, float gamma, int c_reg,int* nprobes);
#endif