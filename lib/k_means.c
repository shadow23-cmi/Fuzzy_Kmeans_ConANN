#include "k_means.h"

float RandomFfloat(float min, float max)
{
    return min + ((float)rand() / (float)RAND_MAX) * (max - min);
}

float* read_fvecs(const char* filename, int* num_vectors, int* dimension) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    // Read first dimension
    int dim;
    if (fread(&dim, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    *dimension = dim;

    // Get file size
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);

    // Each vector:
    // int dimension + float[dimension]
    long vec_size = sizeof(int) + dim * sizeof(float);

    *num_vectors = filesize / vec_size;

    // Allocate memory
    float* data = (float*)malloc((*num_vectors) * dim * sizeof(float));
    if (!data) {
        fclose(fp);
        return NULL;
    }

    // Read vectors
    rewind(fp);

    for (int i = 0; i < *num_vectors; i++) {
        int d;

        fread(&d, sizeof(int), 1, fp);

        if (d != dim) {
            fprintf(stderr, "Dimension mismatch\n");
            free(data);
            fclose(fp);
            return NULL;
        }

        fread(data + i * dim, sizeof(float), dim, fp);
    }

    fclose(fp);
    return data;
}

int* read_ivecs(const char* filename, int* num_vectors, int* dimension) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return NULL;
    }

    // Read first dimension
    int dim;
    if (fread(&dim, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return NULL;
    }

    *dimension = dim;

    // Get file size
    fseek(fp, 0, SEEK_END);
    long filesize = ftell(fp);

    // Each vector:
    // int dimension + int[dimension]
    long vec_size = sizeof(int) + dim * sizeof(int);

    *num_vectors = filesize / vec_size;

    // Allocate memory
    int* data = (int*)malloc((*num_vectors) * dim * sizeof(int));
    if (!data) {
        fclose(fp);
        return NULL;
    }

    // Read vectors
    rewind(fp);

    for (int i = 0; i < *num_vectors; i++) {
        int d;

        fread(&d, sizeof(int), 1, fp);

        if (d != dim) {
            fprintf(stderr, "Dimension mismatch\n");
            free(data);
            fclose(fp);
            return NULL;
        }

        fread(data + i * dim, sizeof(int), dim, fp);
    }

    fclose(fp);
    return data;
}


DATA_FLOAT * populate_flat_indexed_data_float(DATA_FLOAT * dataset, float* data, int dim, int no_of_data_pts)
{
    /* Cretae Data type dataset from plain array*/
    dataset->data = data;
    dataset->dim = dim;
    dataset->no_of_data_pts = no_of_data_pts;
    dataset->shape[0] = no_of_data_pts;
    dataset->shape[1] = dim;
    dataset->labels = (int*)malloc(no_of_data_pts*sizeof(int));
    
    return dataset;
}

DATA_INT * populate_flat_indexed_data_int(DATA_INT* dataset, int* data, int dim, int no_of_data_pts)
{
    /* Cretae Data type dataset from plain array*/
    dataset->data = data;
    dataset->dim = dim;
    dataset->no_of_data_pts = no_of_data_pts;
    dataset->shape[0] = no_of_data_pts;
    dataset->shape[1] = dim;
    dataset->labels = (int*)malloc(no_of_data_pts*sizeof(int));
    
    return dataset;
}

float inline l2_distance(float* a, float* b, int dim)
{
    float dist = 0,temp;
    int i;
    for (i=0;i<dim;i++)
    {
        temp = a[i]-b[i];
        dist += (temp * temp);
    }
    return dist;
}


/*
float l2_distance_parallel(float* a, float* b, int dim)
{
    float dist = 0.0f;
    
    #pragma omp parallel for reduction(+:dist)
    for (int i = 0; i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    
    return dist;
}
*/


float l2_distance_avx2(float* a, float* b, int dim)
{
    __m256 sum = _mm256_setzero_ps();
    const int simd_width = 8;
    
    for (int i = 0; i < dim - simd_width; i += simd_width) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 diff = _mm256_sub_ps(va, vb);
        __m256 squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
    }
    
    // Horizontal sum
    float temp[8];
    _mm256_storeu_ps(temp, sum);
    float dist = temp[0] + temp[1] + temp[2] + temp[3] + 
                 temp[4] + temp[5] + temp[6] + temp[7];
    
    // Handle remaining elements
    for (int i = dim - (dim % simd_width); i < dim; i++) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    
    return dist;
}


void initialize_centroids(DATA_FLOAT  *dataset, CLUSTERS *cls, int num_clusters)
{
    int i;
    int dim = dataset->dim;
    //cls = malloc(sizeof(CLUSTERS));
    //cls->clusters = malloc(num_clusters*sizeof(CLUSTER));
    cls->num_clusters = num_clusters;
    cls->dim = dim;
    cls->centroids = (float*)calloc(num_clusters*dim,sizeof(float));
    cls->cluster_sizes = (int*)calloc(num_clusters,sizeof(int));

    //float min_val = DatasetMin(dataset);
    //float max_val = DatasetMax(dataset);

    for(i=0;i<num_clusters;i++)
    {
        cls->cluster_sizes[i] = 0;
    }
    srand(time(NULL));
    //srand(1234);

    for(i=0;i<num_clusters;i++)
    {
        int idx = rand() % dataset->no_of_data_pts;

        memcpy(&cls->centroids[i*dim],
                &dataset->data[idx*dim],
                dim*sizeof(float));
    }
}

void update_centroids(DATA_FLOAT  *dataset, CLUSTERS *cls)
{
    int i, j;
    int label;

    int num_clusters = cls->num_clusters;
    int dim = dataset->dim;
    int n = dataset->no_of_data_pts;

    // STEP 1: reset centroids + sizes
    for(i = 0; i < num_clusters; i++)
    {
        cls->cluster_sizes[i] = 0;
        for(j = 0; j < dim; j++)
            cls->centroids[i*dim+j] = 0.0f;
    }

    // STEP 2: accumulate
    for(i = 0; i < n; i++)
    {
        label = dataset->labels[i];
        cls->cluster_sizes[label]++;

        for(j = 0; j < dim; j++)
        {
            cls->centroids[label*dim + j] +=
                dataset->data[i * dim + j];
        }
    }

    // STEP 3: normalize
    for(i = 0; i < num_clusters; i++)
    {
        if(cls->cluster_sizes[i] > 0)
        {
            for(j = 0; j < dim; j++)
                cls->centroids[i*dim + j] /= cls->cluster_sizes[i];
        }
    }
}

int assign_clusters(DATA_FLOAT  *dataset, CLUSTERS *cls)
{
    int i, j;
    //int old_label, new_label;
    int no_of_assignment_changes = 0;

    int num_clusters = cls->num_clusters;
    int dim = dataset->dim;
    int no_of_data_pts = dataset->no_of_data_pts;

    // parallelize the loop but keeps shared vasriable no_of_assignment_changes thread safe 
    #pragma omp parallel for reduction(+:no_of_assignment_changes)
    for(i = 0; i < no_of_data_pts; i++)
    {
        // ***very importatnt*** that thse variables stays local to each loop
        // proper parallelization
        int old_label = dataset->labels[i];
        int new_label = 0;

        float best_dist = l2_distance(&dataset->data[i * dim], &cls->centroids[0], dim);

        for(j = 1; j < num_clusters; j++)
        {
            float d = l2_distance(&dataset->data[i * dim], &cls->centroids[j*dim], dim);
            if(d < best_dist)
            {
                best_dist = d;
                new_label = j;
            }
        }
        dataset->labels[i] = new_label;

        if(new_label != old_label)
            no_of_assignment_changes++;
    }

    return no_of_assignment_changes;
}

void train_kmeans(DATA_FLOAT  *train_dataset, CLUSTERS *cls)
{
    int iter;
    int changes;
    float prev_sse = 1e30, sse, sse_improvement;
    int no_of_data_pts = train_dataset->no_of_data_pts;
    double start = omp_get_wtime();

    int max_iters = (train_dataset->dim) * (train_dataset->no_of_data_pts);
    /* Initialize labels */
    for(int i = 0; i < train_dataset->no_of_data_pts; i++)
        train_dataset->labels[i] = 0;

    /* Initial assignment */
    for(int i = 0; i < cls->num_clusters; i++)
        cls->cluster_sizes[i]= 0;

    for(int i = 0; i < train_dataset->no_of_data_pts; i++)
        cls->cluster_sizes[0]++;

    //cls->train_data = train_dataset;
    for(iter = 0; iter < max_iters; iter++)
    {
        changes = assign_clusters(train_dataset, cls);
        update_centroids(train_dataset, cls);

        sse = compute_sse(train_dataset, cls);
        sse_improvement = (prev_sse - sse) / prev_sse;

        printf("Iteration %d: %d assignment changes, SSE = %0.2f\n", iter, changes, sse);
        if(((float)changes / (float)no_of_data_pts) <= 0.005f || sse_improvement <= 1e-4f)
        {
            printf("Converged in %d iterations\n", iter + 1);
            break;
        }
        prev_sse = sse;
    }
    double end = omp_get_wtime();
    printf("Total training time = %.2f sec\n", end - start);
}

float compute_sse(DATA_FLOAT  *dataset, CLUSTERS *cls)
{
    float sse = 0.0f;
    int dim = dataset->dim;
    #pragma omp parallel for reduction(+:sse)
    for(int i=0;i<dataset->no_of_data_pts;i++)
    {
        int label = dataset->labels[i];

        sse += l2_distance(&dataset->data[i*dataset->dim],
                            &cls->centroids[label*dim],
                            dataset->dim);
    }

    return sse;
}

int* assign_fuzzy_clusters(DATA_FLOAT  *dataset, CLUSTERS *cls)
{
    int i, j;
    //int old_label, new_label;
    int no_of_assignment_changes = 0;

    int num_clusters = cls->num_clusters;
    int dim = dataset->dim;
    int no_of_data_pts = dataset->no_of_data_pts;
    int* second_labels = (int*)malloc(no_of_data_pts*sizeof(int));

    // parallelize the loop but keeps shared vasriable no_of_assignment_changes thread safe 
    #pragma omp parallel for reduction(+:no_of_assignment_changes)
    for(i = 0; i < no_of_data_pts; i++)
    {
        // ***very importatnt*** that thse variables stays local to each loop
        // proper parallelization
        int old_label = dataset->labels[i];
        int new_label = 0;
        int new_2nd_label = -1;

        float best_dist = l2_distance(&dataset->data[i * dim], &cls->centroids[0], dim);
        //float second_best_dist = l2_distance(&dataset->data[i * dim], &cls->centroids[0], dim);
        float second_best_dist  = FLT_MAX;
        /*
        if(second_best_dist>best_dist)
        {
            float temp = best_dist;
            best_dist = second_best_dist;
            second_best_dist = temp;

            int temp_label = new_label;
            new_label = new_2nd_label;
            new_2nd_label = temp_label;
        }*/

        for(j = 1; j < num_clusters; j++)
        {
            float d = l2_distance(&dataset->data[i * dim], &cls->centroids[j*dim], dim);
            if(d < best_dist)
            {
                second_best_dist = best_dist;
                new_2nd_label = new_label;

                best_dist = d;
                new_label = j;
            }
            else if(d < second_best_dist)
            {
                second_best_dist = d;
                new_2nd_label = j;
            }
        }
        dataset->labels[i] = new_label;
        //second_labels[i] = new_2nd_label;
        //second_labels[i] = (second_best_dist/best_dist) > 0.9 ? new_2nd_label : -1;
        // Handle case where no second best exists
        if(new_2nd_label != -1) {
            float ratio = best_dist / second_best_dist;
            second_labels[i] = (ratio > 0.97) ? new_2nd_label : -1;
        } else {
            second_labels[i] = -1; // No second cluster available
        }

        if(new_label != old_label)
            no_of_assignment_changes++;
    }
    return second_labels;
}