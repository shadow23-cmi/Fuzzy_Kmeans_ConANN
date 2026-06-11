#include "k_means.h"
#include "IVF_ANN.h"
#include "ConANN.h"
#include <math.h>

/*
void static printInt(const void *p) {
    printf("%d", *(const int *)p);
}
*/
void static printFloat(const void *p) {
    printf("%.2f", *(const float *)p);
}

int main(void)
{
    int i;
    int n_xt, dim_xt; // number of data pts and dimension of training dataset
    int n_xb, dim_xb; // number of data pts and dimension of search database dataset
    int n_xq, dim_xq; // number of data pts and dimension of query dataset
    int n_xc, dim_xc; // number of data pts and dimension of calibration dataset
    int n_gt, dim_gt; // number of data pts and dimension of query groundtruth dataset
    int n_gt_xc, dim_gt_xc; // number of data pts and dimension of calibration groundtruth dataset
    int nlist = 1000;
    int k = 10;
    float D_max, D_min;
    srand(time(NULL));
    //int* indices = NULL;
    int* nprobes;
    int c_reg = 4;
    float gamma=0.01;//alpha=0.44;
    float alphas[10] = {0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5};
    float emperical_fnr[10];
    int num_alphas = 10;
    //float* distances = NULL;
    float* thresholds = NULL;
    //SEARCH_RESULT* KNN;
    SEARCH_RESULTS* results;

    DATA_FLOAT  xt,xb,xq,xc; // dataset for training database and query
    DATA_INT gt,gt_xc;

    printf("Reading and preparing Datasets....\n");
    float* training_vectors = read_fvecs("sift/sift_learn.fvecs", &n_xt, &dim_xt);
    float* database_vectors = read_fvecs("sift/sift_base.fvecs", &n_xb, &dim_xb);
    float* calibration_vectors = read_fvecs("sift/sift_calibration.fvecs", &n_xc, &dim_xc);
    float* query_vectors = read_fvecs("sift/sift_query.fvecs", &n_xq, &dim_xq);
    int* ground_truth_vectors = read_ivecs("sift/sift_groundtruth.ivecs", &n_gt, &dim_gt);
    int* calibration_ground_truth_vectors = read_ivecs("sift/sift_calibration_groundtruth.ivecs", &n_gt_xc, &dim_gt_xc);
    //printf("check2\n");
    populate_flat_indexed_data_float(&xt, training_vectors, dim_xt, n_xt);
    populate_flat_indexed_data_float(&xb, database_vectors, dim_xb, n_xb);
    populate_flat_indexed_data_float(&xq, query_vectors, dim_xq, n_xq);
    populate_flat_indexed_data_float(&xc, calibration_vectors, dim_xc, n_xc);
    populate_flat_indexed_data_int(&gt, ground_truth_vectors, dim_gt, n_gt);
    populate_flat_indexed_data_int(&gt_xc, calibration_ground_truth_vectors, dim_gt_xc, n_gt_xc);
    //printf("check3\n");
    IVF_Index * index = InitIVFIndex(&xt,dim_xt,nlist);
    //printf("check4\n");
    TrainIVFIndex(index);
    //AddDataBase(index,&xb);
    AddDataBaseFuzzy(index, &xb);
    for(int i=0;i<100;i++) printf("-");
    printf("\n");
    printf("Calibrating ANN...\n");
    thresholds = Calibrartion(index,&xc,&gt_xc,&D_min,&D_max,k,alphas,num_alphas,gamma,c_reg);

    /*Output for the whole query dataset with varying target fnrs..*/
    for(int i=0;i<100;i++) printf("-");
    printf("\n");
    printf("Output for whole query dataset with varying target fnr:\n");
    printf("gamma = %f, c_reg = %d\n",gamma,c_reg);
    for(i=0;i<num_alphas;i++)
    {
        nprobes = (int*)malloc(n_xq*sizeof(int));
        double start = omp_get_wtime();
        results = SearchMultipleQueriesAdaptive(index,&xq,D_min,D_max,thresholds[i],k,gamma,c_reg,nprobes);
        double end = omp_get_wtime();
        printf("At k=%d for target fnr = %f,threshold = %f ,emperical fnr = %0.5f, average nprobe = %d time: %.2f secs\n", 
                        k, alphas[i],thresholds[i],1.0f-RecallMultipleQuery(results->idx,gt.data,
                            (int)(gt.no_of_data_pts),k),(int)Mean(nprobes,n_gt)+1,end - start);
        emperical_fnr[i] = 1.0f-RecallMultipleQuery(results->idx,gt.data,
                            (int)(gt.no_of_data_pts),k);
        free(nprobes);
        free(results);
    }
    printf("alphas:           ");
    printArray(&alphas,num_alphas,sizeof(float),printFloat);
    printf("learned threshold:");
    printArray(thresholds,num_alphas,sizeof(float),printFloat);
    printf("emperical_fnr:    ");
    printArray(&emperical_fnr,num_alphas,sizeof(float),printFloat);
    
    
    return 0;
}