#include "k_means.h"
#include "IVF_ANN.h"
#include <math.h>

void static printInt(const void *p) {
    printf("%d", *(const int *)p);
}

void static printFloat(const void *p) {
    printf("%.2f", *(const float *)p);
}



int main(void)
{
    int n_xt, dim_xt; // number of data pts and dimension of training dataset
    int n_xb, dim_xb; // number of data pts and dimension of search database dataset
    int n_xq, dim_xq; // number of data pts and dimension of query dataset
    int n_gt, dim_gt; // number of data pts and dimension of query groundtruth dataset
    int nlist = 1000;
    int nprobe = 31;
    int k = 10;
    srand(time(NULL));
    int query_id = (int)RandomFfloat(0,999);
    int* indices = NULL;
    float* distances = NULL;
    SEARCH_RESULT* KNN;
    SEARCH_RESULTS* results;

    DATA_FLOAT  xt,xb,xq; // dataset for training database and query
    DATA_INT gt;

    printf("Reading and preparing Datasets\n");
    float* training_vectors = read_fvecs("gist/gist_learn.fvecs", &n_xt, &dim_xt);
    float* database_vectors = read_fvecs("gist/gist_base.fvecs", &n_xb, &dim_xb);
    float* query_vectors = read_fvecs("gist/gist_query.fvecs", &n_xq, &dim_xq);
    int* ground_truth_vectors = read_ivecs("gist/gist_groundtruth.ivecs", &n_gt, &dim_gt);
    //printf("check2\n");
    populate_flat_indexed_data_float(&xt, training_vectors, dim_xt, n_xt);
    populate_flat_indexed_data_float(&xb, database_vectors, dim_xb, n_xb);
    populate_flat_indexed_data_float(&xq, query_vectors, dim_xq, n_xq);
    populate_flat_indexed_data_int(&gt, ground_truth_vectors, dim_gt, n_gt);
    //printf("check3\n");
    IVF_Index * index = InitIVFIndex(&xt,dim_xt,nlist);
    //printf("check4\n");
    TrainIVFIndex(index);
    AddDataBase(index,&xb);
    //AddDataBaseFuzzy(index, &xb);
    for(int i=0;i<100;i++) printf("-");
    printf("\n");
    /*Output for a single query search*/
    printf("Output for a random single query search:\n");
    printf("Query Vector:\n");
    for (int j = 0; j < dim_xq; j++) {
        printf("%.2f ", query_vectors[query_id*dim_xq+j]);
    }
    printf("\n");
    KNN = SearchSingleQuery(index,&xq,query_id,nprobe,k);
    printf("check6\n");
    SearchResultToArray(KNN,&distances,&indices,k);
    //printf("check7\n");
    printf("\nDistances:");
    printArray(distances,k,sizeof(float),printFloat);
    printf("\nIndices:");
    printArray(indices,k,sizeof(int),printInt);
    printf("\nGround Truths:");
    printArray(&gt.data[query_id*100],k,sizeof(int),printInt);
    //printf("check8\n");
    printf("\nRecall at k=%d for query_id: %d is %0.2f\n", 
                        k, query_id, RecallSingleQuery(indices,&gt.data[query_id*dim_gt],k));
    //printf("check9\n");
    
    /*Output for the whole query dataset with varying nprobes 1 to 40*/
    for(int i=0;i<100;i++) printf("-");
    printf("\n");
    printf("Output for whole query dataset:\n");
    for(int i=1;i<=40;i++)
    {
        double start = omp_get_wtime();
        results = SearchMultipleQueries(index,&xq,i,k);
        double end = omp_get_wtime();
        printf("Recall at k=%d for nprobe %d is %0.5f, time: %.2f secs\n", 
                        k, i,RecallMultipleQuery(results->idx,gt.data,(int)(gt.no_of_data_pts),k),end - start);
    }
    return 0;
}