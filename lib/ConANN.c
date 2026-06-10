#include "IVF_ANN.h"
#include "ConANN.h"

/*
int compare_search_results(const void* a, const void* b)
{
    SEARCH_RESULT* res1 = (SEARCH_RESULT*)a;
    SEARCH_RESULT* res2 = (SEARCH_RESULT*)b;

    if(res1->dist > res2->dist) return 1;
    else if(res1->dist == res2->dist) return 0;
    else return -1;
}
    
int compare_search_results(const void *a, const void *b)
{
    const SEARCH_RESULT *r1 = (const SEARCH_RESULT *)a;
    const SEARCH_RESULT *r2 = (const SEARCH_RESULT *)b;

    if (r1->dist < r2->dist)
        return -1;

    if (r1->dist > r2->dist)
        return 1;

    // tie-break using index 
    if (r1->idx < r2->idx)
        return -1;

    if (r1->idx > r2->idx)
        return 1;

    return 0;
}
*/
float inline l2_distance_norm(float* a, float* b, int dim)
{
    float* zero_vec = (float*)calloc(dim,sizeof(float));
    float a_norm = l2_distance(a,zero_vec,dim);
    float b_norm = l2_distance(b,zero_vec,dim);
    float dist = 0,temp;
    int i;
    for (i=0;i<dim;i++)
    {
        temp = (a[i]/a_norm)-(b[i]/b_norm);
        dist += (temp * temp);
    }
    free(zero_vec);
    return dist;
}

float l2_norm(float* a, int dim)
{
    float* zero_vec = (float*)calloc(dim,sizeof(float));
    float a_norm = (float)l2_distance(a,zero_vec,dim);
    free(zero_vec);
    return a_norm;
}

static float* DatasetMin(DATA_FLOAT * dataset)
{
    int i,j;
    int dim = dataset->dim;
    int no_of_data_pts = dataset->no_of_data_pts;
    float* min = (float*)malloc(dim*sizeof(float));
    for(i=0;i<dim;i++)
        min[i] = dataset->data[i];
    for(i=0;i<dim;i++)
    {
        for(j=0;j<no_of_data_pts;j++)
            min[i] = min[i] < (dataset->data[j*dim+i]) ? min[i] : (dataset->data[j*dim+i]);
    }
    return min;
}

static float* DatasetMax(DATA_FLOAT * dataset)
{
    int i,j;
    int dim = dataset->dim;
    int no_of_data_pts = dataset->no_of_data_pts;
    float* max = (float*)malloc(dim*sizeof(float));
    for(i=0;i<dim;i++)
        max[i] = dataset->data[i];
    for(i=0;i<dim;i++)
    {
        for(j=0;j<no_of_data_pts;j++)
            max[i] = max[i] > (dataset->data[j*dim+i]) ? max[i] : (dataset->data[j*dim+i]);
    }
    return max;
}

float GlobalMaxDistanceL2(DATA_FLOAT* dataset)
{
    int dim = dataset->dim;
    float* geometric_min_data_pt = DatasetMin(dataset);
    float* geometric_max_data_pt = DatasetMax(dataset);
    return sqrtf(l2_distance(geometric_min_data_pt, geometric_max_data_pt, dim));
}
/*
float* ComputeScoreSingleQuery(IVF_Index *index, DATA_FLOAT *calibration_dataset, int query_id, int k)
{
    int i,j;
    int nlist = index->cls->num_clusters;
    int P = nlist;
    int dim = index->cls->dim;
    int bst_root = -1;
    int bst_max_element_pos;
    int exists_in_bst = -1;
    //int threshold_achives = 0;
    float kth_dist;
    BST top_k_results;
    //SEARCH_RESULT* results;
    
    SEARCH_RESULT result;
    SEARCH_RESULT* kth_result;
    //SEARCH_RESULT* results_k = NULL;

    SEARCH_RESULT* clusters = DistancesToClusters(index,calibration_dataset,query_id);
    InitBST(&top_k_results, sizeof(SEARCH_RESULT), k, compare_search_results);
    float* single_query_scores = (float*)malloc(P*sizeof(float));
    for(i=0;i<P;i++)
    {
        int cluster_id = clusters[i].idx;
        int cluster_size = index->inverted_lists_length[cluster_id];
        //results = SearchClusterAdaptive(index, query_database, cluster_id, query_id, k);
        for(j=0;j<cluster_size;j++)
        {
            int idx = index->inverted_lists[cluster_id][j];
            float dist = l2_distance(&index->search_database->data[idx*dim],&calibration_dataset->data[query_id*dim],dim);
            result.idx = idx;
            result.dist = dist;
            exists_in_bst = Search(&top_k_results,bst_root,&result);
            if(top_k_results.no_elements<k && exists_in_bst==-1)
            {
                bst_root = Insert(&top_k_results,bst_root,&result);
            }
            else if(top_k_results.no_elements>=k)
            {
                bst_max_element_pos = FindMax(&top_k_results,bst_root);
                kth_result = (SEARCH_RESULT*)DATA_AT(&top_k_results,bst_max_element_pos);
                kth_dist = kth_result->dist;
                if((top_k_results.comparator(&result,kth_result)<0)
                    && (exists_in_bst==-1))
                {
                    bst_root = Delete(&top_k_results,bst_root,kth_result);
                    bst_root = Insert(&top_k_results,bst_root,&result);
                }
                
            }
            
        }
        bst_max_element_pos = FindMax(&top_k_results,bst_root);
        kth_result = (SEARCH_RESULT*)DATA_AT(&top_k_results,bst_max_element_pos);
        kth_dist = kth_result->dist;
        single_query_scores[i] = sqrt(kth_dist);
        
    }
    
    return single_query_scores;
}
*/
/*
float* ComputeScoreSingleQuery(IVF_Index *index, DATA_FLOAT *calibration_dataset, DATA_INT* calibration_groundtruth,float* recalls,int query_id, int k)
{
    int i;
    int nlist = index->cls->num_clusters;
    int P = nlist;
    float *local_distances = NULL;
    int *local_indices = NULL;

    SEARCH_RESULT* clusters = DistancesToClusters(index,calibration_dataset,query_id);
    SEARCH_RESULT* results = (SEARCH_RESULT*)calloc(2*k,sizeof(SEARCH_RESULT));
    float* single_query_scores = (float*)malloc(P*sizeof(float));
    
    SEARCH_RESULT* temp = SearchCluster(index,calibration_dataset,clusters[0].idx,query_id,k);
    memcpy(&results[0],temp,k*sizeof(SEARCH_RESULT));
    qsort(results, k, sizeof(SEARCH_RESULT), compare_search_results);
    single_query_scores[0] = sqrtf(results[k-1].dist);
    SearchResultToArray(results, &local_distances, &local_indices, k);
    recalls[query_id*P+0] = RecallSingleQuery(local_indices, &calibration_groundtruth->data[query_id*100], k);
    if(query_id==0)
            printf("query %d p=%d kth=%f\n",query_id, 1, results[k-1].dist);
    free(local_distances);
    free(local_indices);
    //single_query_scores[0] = results[k-1].dist>0?sqrtf(results[k-1].dist):0;
    //#pragma omp parallel for
    for(i=1;i<P;i++)
    {
        SEARCH_RESULT* temp = SearchCluster(index,calibration_dataset,clusters[i].idx,query_id,k);
        memcpy(&results[k],temp,k*sizeof(SEARCH_RESULT));
        qsort(results, 2*k, sizeof(SEARCH_RESULT), compare_search_results);
        free(temp);
        single_query_scores[i] = sqrtf(results[k-1].dist);
        //single_query_scores[i] = results[k-1].dist;
        SearchResultToArray(results, &local_distances, &local_indices, k);
        recalls[query_id*P+i] = RecallSingleQuery(local_indices, &calibration_groundtruth->data[query_id*100], k);
        free(local_distances);
        free(local_indices);
        if(query_id==0)
            printf("query %d p=%d kth=%f\n",query_id, i+1, results[k-1].dist);
    }
    return single_query_scores;
}
*/

float* ComputeScoreSingleQuery(IVF_Index *index, DATA_FLOAT *calibration_dataset, DATA_INT* calibration_groundtruth,
                                float* recalls,int query_id, int P,int k)
{
    int i;
    //int nlist = index->cls->num_clusters;
    //int P = nlist;
    float* single_query_scores = (float*)malloc(P*sizeof(float));
    SEARCH_RESULT* clusters = DistancesToClusters(index,calibration_dataset,query_id);
    SEARCH_RESULT* results = (SEARCH_RESULT*)malloc(P*k*sizeof(SEARCH_RESULT));
    //SEARCH_RESULT* results_k = (SEARCH_RESULT*)malloc(k*sizeof(SEARCH_RESULT));
    //printf("\nSearching %d nearest neighbours for query_id: %d....\n",k, query_id);
    #pragma omp parallel for
    for(i=0;i<P;i++)
    {
        SEARCH_RESULT* temp = SearchCluster(index,calibration_dataset,clusters[i].idx,query_id,k);
        memcpy(&results[i*k],temp,k*sizeof(SEARCH_RESULT));
        free(temp);
    }
    for(i=0;i<P;i++)
    {
        float *local_distances = NULL;
        int *local_indices = NULL;
        qsort(results, i*k+k, sizeof(SEARCH_RESULT), compare_search_results);
        //if(i>=2) memcpy(&results[k],&results[i*k],k*sizeof(SEARCH_RESULT));
        //qsort(results, 2*k, sizeof(SEARCH_RESULT), compare_search_results);
        single_query_scores[i] = sqrtf(results[k-1].dist);
        SearchResultToArray(results, &local_distances, &local_indices, k);
        recalls[query_id*P+i] = RecallSingleQuery(local_indices, &calibration_groundtruth->data[query_id*100], k);
        free(local_distances);
        free(local_indices);
        //if(query_id==50)
        //    printf("query %d p=%d kth=%f, recall: %f\n",query_id, i+1, results[k-1].dist,recalls[query_id*P+i]);
    }
    
    free(results);
    free(clusters);
    //resturn only k closest results
    return single_query_scores;
}

float** ComputeScore(IVF_Index *index, DATA_FLOAT *calibration_dataset, DATA_INT* calibration_groundtruth,
                    float* recalls,int P,int k)
{
    int i,j=0;
    int num_queries = calibration_dataset->no_of_data_pts;
    int M = num_queries;
    
    float** scores = (float**)malloc(M*sizeof(float*));
    
    //#pragma omp parallel for schedule(dynamic)
    for(i=0;i<M;i++)
    {
        scores[i] = ComputeScoreSingleQuery(index,calibration_dataset,calibration_groundtruth,recalls,i,P,k);
        if(i == j*M/10)
        {
            j++;
            printf("score computed for %d %% queries\n",j*10);
        }   
    }
    
    return scores;
}

float* Calibrartion(IVF_Index *index, DATA_FLOAT *calibration_dataset, DATA_INT* calibration_groundtruth,
                     float* D_min,float* D_max,int k, float* alphas, int num_alphas,float gamma, int c_reg)
{
    int i,j,m;
    //int dim = calibration_dataset->dim;
    int num_queries = calibration_dataset->no_of_data_pts;
    int M = num_queries;
    int nlist = index->cls->num_clusters;
    int P = nlist/6;
    float float_M = (float)M;
    double start = omp_get_wtime();
    float** r = (float**)malloc(100*sizeof(float*));
    float* recalls = (float*)malloc(P*M*sizeof(float));
    float* best_lambdas = (float*)malloc(num_alphas*sizeof(float));
    // line 1 - 8 of calibration algorithm
    float global_max_norm = GlobalMaxDistanceL2(index->search_database);
    float** scores = ComputeScore(index, calibration_dataset, calibration_groundtruth,recalls,P,k);
    printf("global_max_norm = %f\n",global_max_norm);
    //printf("sample kth dist = %f\n",ComputeScore(index, calibration_dataset, k)[0][0]);
    float min_raw = 1e30f;
    float max_raw = -1.0f;
    for(i=0;i<M;i++)
    {
        for(int p=0;p<P;p++)
        {
            if(!isfinite(scores[i][p]))
        {
            printf("BAD SCORE: i=%d p=%d score=%f\n",
                   i,p,scores[i][p]);
            exit(1);
        }
             if(scores[i][p] < min_raw)
            min_raw = scores[i][p];

            if(scores[i][p] > max_raw)
            max_raw = scores[i][p];
        }
    }
    *D_min = min_raw;
    *D_max = max_raw;
    printf("raw range = [%f, %f]\n", min_raw, max_raw);
    #pragma omp parallel for schedule(dynamic)
    for(i=0; i<M; i++)
    {
        for(int p=0; p<P; p++)
        {
            float score = scores[i][p];
            //score = score / global_max_norm;
            score = (score-min_raw)/(max_raw-min_raw);
            score = 1.0f - score;
            score += gamma * MAX(0, p - c_reg);
            scores[i][p] = score;
        }

        if(i % (M/10) == 0)
        {
            #pragma omp critical
            {
                printf("processed %d / %d queries\n", i, M);
            }
        }
    }

    float min_score = scores[0][0];
    float max_score = scores[0][0];

    for(i=0;i<M;i++)
    {
        for(int p=0;p<P;p++)
        {
            if(scores[i][p] < min_score)
                min_score = scores[i][p];
            if(scores[i][p] > max_score)
                max_score = scores[i][p];
        }
    }

    printf("score range = [%f, %f]\n", min_score, max_score);
    //printf("Query 0:\n");
    //for(int p=0;p<20;p++) printf("%d %.6f", p+1, scores[0][p]);
    //printf("\n Query 100:\n");
    //for(int p=0;p<20;p++) printf("%d %.6f", p+1, scores[100][p]);
    printf("\n");
    //line 9 - 15 of the same algorithm
    for(m=0;m<num_alphas;m++)
    {
    float best_lambda = -1.0f;
    for(j=0; j<101; j++)
    {
        double R = 0.0f;double avg_recall = 0.0;
        float l = (float)j / 100.0f;
        float lambda = 1.0f-l;
        long long total_p = 0;
        r[j] = (float*)malloc(M * sizeof(float));
        //#pragma omp parallel for schedule(dynamic)
        #pragma omp parallel for reduction(+:avg_recall,total_p,R)
        for(i=0; i<M; i++)
        {
            int p_dash = 0;
            for(int p=0; p<P; p++)
            {
                if(scores[i][p] >= lambda)
                {
                    p_dash = p;
                    total_p += p_dash;
                    break;
                }
            }
        
            //printf("\n");
            float recall = recalls[i*P+p_dash];
            avg_recall += recall;

            r[j][i] = (1.0f - recall) - (alphas[m] * (float_M + 1.0f) / float_M) - (1.0f / float_M);
            R += r[j][i];
        }
        R /= float_M;
        avg_recall /= float_M;
        if(R > 0)
        {
            printf("target fnr = %f,lambda=%f R = %f avg recall = %f avg_p=%f\n", alphas[m],lambda, R,avg_recall,(double)total_p/M);
            best_lambda = lambda;
            break;
        }
        }
        best_lambdas[m] = best_lambda;
    }

    double end = omp_get_wtime();
    printf("\nCalibration done in %.2f secs\n",end-start);
    return best_lambdas;
}


SEARCH_RESULT *SearchSingleQueryAdaptive(IVF_Index *index, DATA_FLOAT *query_database,
                                         int query_id, float D_min,float D_max,float threshold, int k, float gamma, int c_reg,int* nprobe)
{
    int p;
    int nlist = index->cls->num_clusters;
    *nprobe = nlist;
    //float D_max = 1964.0f;
    SEARCH_RESULT* clusters = DistancesToClusters(index,query_database,query_id);
    SEARCH_RESULT* results = (SEARCH_RESULT*)malloc(2*k*sizeof(SEARCH_RESULT));
    SEARCH_RESULT* results_k = (SEARCH_RESULT*)malloc(k*sizeof(SEARCH_RESULT));
    //printf("\nSearching %d nearest neighbours for query_id: %d....\n",k, query_id);
    SEARCH_RESULT* temp = SearchCluster(index,query_database,clusters[0].idx,query_id,k);
    memcpy(&results[0],temp,k*sizeof(SEARCH_RESULT));
    qsort(results, k, sizeof(SEARCH_RESULT), compare_search_results);
    //#pragma omp parallel for
    for(p=1;p<nlist;p++)
    {
        SEARCH_RESULT* temp = SearchCluster(index,query_database,clusters[p].idx,query_id,k);
        memcpy(&results[k],temp,k*sizeof(SEARCH_RESULT));
        qsort(results, 2*k, sizeof(SEARCH_RESULT), compare_search_results);
        free(temp);
        //if(results[k-1].dist<=threshold)
        float score = sqrtf((float)results[k-1].dist);
        //score = (score/(float)D_max);
        score = (score-D_min)/(D_max-D_min);
        score = 1.0f - score;
        score += gamma * MAX(0, p - c_reg);
        if(score>=(threshold))
        {
            *nprobe=p+1;break;
        }
    }
    //memcpy(results_k,results,k*sizeof(SEARCH_RESULT));
    results_k[0].dist = results[0].dist;
    results_k[0].idx = results[0].idx;
    
    for(p=1;p<k;p++)
    {
        results_k[p].dist = results[p].dist;
        results_k[p].idx = results[p].idx;
    }
    //printf("searched done for query %d\n",query_id);
    free(results);
    free(clusters);
    //resturn only k closest results
    return results_k;
}

SEARCH_RESULTS *SearchMultipleQueriesAdaptive(IVF_Index *index, DATA_FLOAT *query_database,
                                                 float D_min,float D_max,float threshold, int k, float gamma, int c_reg,int* nprobes)
{
    int i,j;
    int num_queries = query_database->no_of_data_pts;
    //float D_max = GlobalMaxDistanceL2(index->search_database);
    SEARCH_RESULT* result[num_queries];
    SEARCH_RESULTS* results = (SEARCH_RESULTS*)malloc(sizeof(SEARCH_RESULTS));
    results->k = k;
    results->num_queries = num_queries;
    float* distances = (float*)malloc(num_queries*k*sizeof(float));
    int* indices = (int*)malloc(num_queries*k*sizeof(int));
    //printf("raw range = [%f, %f]\n",D_min,D_max);
    #pragma omp parallel for schedule(dynamic)
    for(i=0;i<num_queries;i++)
    {
        result[i] = SearchSingleQueryAdaptive(index,query_database,i,D_min,D_max,threshold,k, gamma, c_reg,&nprobes[i]);
        for(j=0;j<k;j++)
        {
            distances[i*k+j] = result[i][j].dist;
            indices[i*k+j] = result[i][j].idx; 
        }
        //printf("query id %d, sreached for threshold %f with nprobes = %d\n",i,threshold,nprobes[i]);
    }
    for(i=0;i<num_queries;i++) free(result[i]);
    results->dist = distances;
    results->idx = indices;
    
    return results;
}
