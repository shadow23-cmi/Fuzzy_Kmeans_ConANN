# include "IVF_ANN.h"

/*
typedef struct{
    CLUSTERS* cls;
    int** inverted_lists;
    DATA_FLOAT * train_data;
    // for training the k_means clustering algorithm
    // usually a portion of search_database;
    DATA_FLOAT * search_database;
} IVF_Index;
*/
int compare_search_results(const void *a, const void *b)
{
    const SEARCH_RESULT *r1 = (const SEARCH_RESULT *)a;
    const SEARCH_RESULT *r2 = (const SEARCH_RESULT *)b;

    if (r1->dist < r2->dist)
        return -1;

    if (r1->dist > r2->dist)
        return 1;

    /* tie-break using index */
    if (r1->idx < r2->idx)
        return -1;

    if (r1->idx > r2->idx)
        return 1;

    return 0;
}

int cmp_search_results(const void* a, const void* b)
{
    SEARCH_RESULT* res1 = (SEARCH_RESULT*)a;
    SEARCH_RESULT* res2 = (SEARCH_RESULT*)b;

    if(res1->dist > res2->dist) return 1;
    else return -1;
}

float Mean(int* array, int num_array)
{
    int i;
    float sum = 0;

    for(i=0;i<num_array;i++)
    {
        sum += (float)array[i];
    }
    sum = sum/num_array;

    return sum;
}

int LinearSearch(int* array, int array_length, int target)
{
    int i;
    for(i=0;i<array_length;i++)
    {
        if(array[i] == target)
            return 1;
    }
    return 0;
}

IVF_Index* InitIVFIndex(DATA_FLOAT * train_data, int dim, int nlist)
{
    IVF_Index* index = NULL;
    CLUSTERS* cls = NULL;

    index = (IVF_Index*)malloc(sizeof(IVF_Index));
    cls = (CLUSTERS*)malloc(sizeof(CLUSTERS));
    //cls->dim = dim;
    //cls->num_clusters = nlist;

    initialize_centroids(train_data, cls, nlist);
    //train_kmeans(train_data, cls);

    index->cls = cls;
    index->train_data = train_data;
    index->inverted_lists_length = (int*)malloc(nlist*sizeof(int));

    return index;
}

void TrainIVFIndex(IVF_Index* index)
{
    printf("\nTraining Clusters....\n");
    train_kmeans(index->train_data, index->cls); 
}

void AddDataBase(IVF_Index* index, DATA_FLOAT * search_database)
{
    int i,label;
    int no_of_data_pts = search_database->no_of_data_pts;
    int num_clusters = index->cls->num_clusters;
    int inverted_lists_last_index[num_clusters];
    
    for(i=0;i<num_clusters;i++) inverted_lists_last_index[i] = 0;
    printf("\nAddiing Search database....\n");
    double start = omp_get_wtime();
    index->search_database = search_database;
    assign_clusters(search_database, index->cls);
    update_centroids(search_database, index->cls);

    index->inverted_lists = (int**)malloc(num_clusters*sizeof(int*));
    for(i = 0; i < num_clusters; i++)
    {
        index->inverted_lists[i] = (int*)malloc((index->cls->cluster_sizes[i]) * sizeof(int));
        index->inverted_lists_length[i] = index->cls->cluster_sizes[i];
    }
    //filling inverted lists from search datbase label of data points
    for(i=0; i<no_of_data_pts;i++)
    {
        label = search_database->labels[i];
        index->inverted_lists[label][inverted_lists_last_index[label]] = i;
        inverted_lists_last_index[label]++;
    }
    double end = omp_get_wtime();
    printf("Search Database added in = %.2f secs\n", end - start);
}

void AddDataBaseFuzzy(IVF_Index* index, DATA_FLOAT * search_database)
{
    int i,label;
    int second_label;
    int no_of_data_pts = search_database->no_of_data_pts;
    int num_clusters = index->cls->num_clusters;
    int inverted_lists_last_index[num_clusters];
    int extra_data_pts_in_cluster[num_clusters],num_extra_pts = 0;
    int* second_labels;
    for(i=0;i<num_clusters;i++) inverted_lists_last_index[i] = 0;
    for(i=0;i<num_clusters;i++) extra_data_pts_in_cluster[i] = 0;
    printf("\nAddiing Search database....\n");
    double start = omp_get_wtime();
    index->search_database = search_database;
    //assign_clusters(search_database, index->cls);
    second_labels = assign_fuzzy_clusters(search_database,index->cls);
    if (!second_labels) {
        fprintf(stderr, "Failed to assign fuzzy clusters\n");
        return;}
    update_centroids(search_database, index->cls);
    for(i=0; i<no_of_data_pts;i++)
        if(second_labels[i] != -1)
            extra_data_pts_in_cluster[second_labels[i]]++;

    index->inverted_lists = (int**)malloc(num_clusters*sizeof(int*));
    if (!(index->inverted_lists)) {
        fprintf(stderr, "Failed to assign fuzzy clusters\n");
        return;}
    for(i = 0; i < num_clusters; i++)
    {
        //index->inverted_lists[i] = (int*)malloc((index->cls->cluster_sizes[i]) * sizeof(int));
        index->inverted_lists[i] = (int*)malloc((index->cls->cluster_sizes[i] + extra_data_pts_in_cluster[i]) * sizeof(int));
        index->inverted_lists_length[i] = index->cls->cluster_sizes[i] + extra_data_pts_in_cluster[i];
        //printf("cluster: %d\n",i);
    }
    //filling inverted lists from search datbase label of data points
    for(i=0; i<no_of_data_pts;i++)
    {
        label = search_database->labels[i];
        index->inverted_lists[label][inverted_lists_last_index[label]] = i;
        inverted_lists_last_index[label]++;
        //printf("%d\n",i);
    }
    for(i=0; i<no_of_data_pts;i++)
    {
        second_label = second_labels[i];
        if(second_label != -1)
        {
            index->inverted_lists[second_label][inverted_lists_last_index[second_label]] = i;
            inverted_lists_last_index[second_label]++;
        }
    }
    for(i=0;i<num_clusters;i++) num_extra_pts += extra_data_pts_in_cluster[i];
    double end = omp_get_wtime();
    free(second_labels);
    printf("Search Database added in = %.2f secs with %.2f %%extra fuzzy points\n", end - start,100.0f*(float)(((float)num_extra_pts)/((float)no_of_data_pts)));
}

SEARCH_RESULT* DistancesToClusters(IVF_Index* index, DATA_FLOAT * query_database, int query_id)
{
    int i;
    //float dist;
    int num_clusters = index->cls->num_clusters;
    int dim = index->train_data->dim;
    SEARCH_RESULT* clusters = NULL;
    clusters = (SEARCH_RESULT*)malloc(index->cls->num_clusters * sizeof(SEARCH_RESULT));
    #pragma omp parallel for
    for(i=0;i<num_clusters;i++)
    {
        float dist = l2_distance(&index->cls->centroids[i*dim],&query_database->data[query_id*dim],dim);
        clusters[i].dist = dist;
        clusters[i].idx = i;
    }

    qsort(clusters, num_clusters,sizeof(SEARCH_RESULT), cmp_search_results);
    return clusters;
}

SEARCH_RESULT* SearchCluster(IVF_Index* index, DATA_FLOAT * query_database, int cluster_id, int query_id, int k)
{
    int i;
    //int idx;
    int dim = index->cls->dim;
    //int cluster_size = index->cls->cluster_sizes[cluster_id];
    int cluster_size = index->inverted_lists_length[cluster_id];
    int min_of_cluster_size_k = k<cluster_size ? k:cluster_size;
    //float dist;

    SEARCH_RESULT* results = (SEARCH_RESULT*)malloc(cluster_size*sizeof(SEARCH_RESULT));
    SEARCH_RESULT* results_k = (SEARCH_RESULT*)malloc(k*sizeof(SEARCH_RESULT));
    for(i=0;i<k;i++)
    {
        results_k[i].dist = FLT_MAX-1;
        results_k[i].idx = -1;
    }
    #pragma omp parallel for
    for(i=0;i<cluster_size;i++)
    {
        int idx = index->inverted_lists[cluster_id][i];
        float dist = l2_distance(&index->search_database->data[idx*dim],&query_database->data[query_id*dim],dim);
        results[i].idx = idx;
        results[i].dist = dist;
    }

    qsort(results, cluster_size, sizeof(SEARCH_RESULT), cmp_search_results);
    memcpy(results_k,results,min_of_cluster_size_k*sizeof(SEARCH_RESULT));
    free(results);
    //resturn only k closest results
    return results_k;
}

SEARCH_RESULT* SearchSingleQuery(IVF_Index* index, DATA_FLOAT * query_database, int query_id, int nprobe, int k)
{
    int i;
    //SEARCH_RESULT* temp = NULL;
    SEARCH_RESULT* clusters = DistancesToClusters(index,query_database,query_id);
    SEARCH_RESULT* results = (SEARCH_RESULT*)malloc(nprobe*k*sizeof(SEARCH_RESULT));
    SEARCH_RESULT* results_k = (SEARCH_RESULT*)malloc(k*sizeof(SEARCH_RESULT));
    //printf("\nSearching %d nearest neighbours for query_id: %d....\n",k, query_id);
    #pragma omp parallel for
    for(i=0;i<nprobe;i++)
    {
        SEARCH_RESULT* temp = SearchCluster(index,query_database,clusters[i].idx,query_id,k);
        memcpy(&results[i*k],temp,k*sizeof(SEARCH_RESULT));
        free(temp);
    }
    qsort(results, nprobe*k, sizeof(SEARCH_RESULT), compare_search_results);
    memcpy(results_k,results,k*sizeof(SEARCH_RESULT));
    
    free(results);
    free(clusters);
    //resturn only k closest results
    return results_k;
}
/*
SEARCH_RESULT *SearchSingleQuery(IVF_Index *index, DATA_FLOAT *query_database, int query_id,  int nprobe, int k)
{
    int i,j;
    int nlist = index->cls->num_clusters;

    //float D_max = GlobalMaxDistanceL2(index->search_database);
    SEARCH_RESULT* clusters = DistancesToClusters(index,query_database,query_id);
    SEARCH_RESULT* results = (SEARCH_RESULT*)malloc(2*k*sizeof(SEARCH_RESULT));
    SEARCH_RESULT* results_k = (SEARCH_RESULT*)malloc(k*sizeof(SEARCH_RESULT));
    //printf("\nSearching %d nearest neighbours for query_id: %d....\n",k, query_id);
    SEARCH_RESULT* temp = SearchCluster(index,query_database,clusters[0].idx,query_id,k);
    memcpy(&results[0],temp,k*sizeof(SEARCH_RESULT));
    qsort(results, k, sizeof(SEARCH_RESULT), compare_search_results);
    #pragma omp parallel for schedule(dynamic)
    for(i=1;i<nprobe;i++)
    {
        SEARCH_RESULT* temp = SearchCluster(index,query_database,clusters[i].idx,query_id,k);
        memcpy(&results[k],temp,k*sizeof(SEARCH_RESULT));
        qsort(results, 2*k, sizeof(SEARCH_RESULT), compare_search_results); // using cmp_search_results here would be dangerous
        free(temp);
    }
    memcpy(results_k,results,k*sizeof(SEARCH_RESULT));
    free(results);
    free(clusters);
    //resturn only k closest results
    return results_k;
}
*/
SEARCH_RESULTS* SearchMultipleQueries(IVF_Index* index, DATA_FLOAT * query_database, int nprobe, int k)
{
    int i,j;
    int num_queries = query_database->no_of_data_pts;
    SEARCH_RESULT* result[num_queries];
    SEARCH_RESULTS* results = (SEARCH_RESULTS*)malloc(sizeof(SEARCH_RESULTS));
    results->k = k;
    results->num_queries = num_queries;
    float* distances = (float*)malloc(num_queries*k*sizeof(float));
    int* indices = (int*)malloc(num_queries*k*sizeof(int));
    #pragma omp parallel for schedule(dynamic)
    for(i=0;i<num_queries;i++)
    {
        result[i] = SearchSingleQuery(index,query_database,i,nprobe,k);
        for(j=0;j<k;j++)
        {
            distances[i*k+j] = result[i][j].dist;
            indices[i*k+j] = result[i][j].idx; 
        }
    }
    for(i=0;i<num_queries;i++) free(result[i]);
    results->dist = distances;
    results->idx = indices;

    return results;
}

float RecallSingleQuery(int* indices, int* ground_truth, int k)
{
    int i;
    float recall = 0;

    for(i=0;i<k;i++)
    {
        recall += (float)LinearSearch(ground_truth, k, indices[i]);
    }
    recall = recall/k;

    return recall;
}

float RecallMultipleQuery(int* indices, int* ground_truth, int num_queries, int k)
{
    int i;
    float recall=0;

    for(i=0;i<num_queries;i++)
    {
        for(int j=0;j<k;j++)
        {
            recall += (float)LinearSearch(&ground_truth[i*100], k, indices[i*k+j]);
        }
    }
    recall = recall/k;
    recall = recall/num_queries;

    return recall;
}

void SearchResultToArray(SEARCH_RESULT* result, float** distances, int** indices,int k)
{
    int i;
    *distances = (float*)malloc(k*sizeof(float));
    *indices = (int*)malloc(k*sizeof(int));
    for(i=0;i<k;i++)
    {
        (*distances)[i] = result[i].dist;
        (*indices)[i] = result[i].idx;
    }
}

void printInt(const void *p) {
    printf("%d", *(const int *)p);
}

void printFloat(const void *p) {
    printf("%.2f", *(const float *)p);
}

void printArray(const void *arr, size_t n, size_t elemSize, void (*printElem)(const void *))
{
    const char *ptr = (const char *)arr;

    printf("[");
    for (size_t i = 0; i < n; i++) {
        printElem(ptr + i * elemSize);

        if (i < n - 1)
            printf(", ");
    }
    printf("]\n");
}
