import faiss
import time
import numpy as np

dataset = "gist"
base_path = dataset + "/" + dataset

search_dataset_extension = "_base.fvecs"
query_dataset_extension = "_query.fvecs"
calibration_dataset_extension = "_calibration.fvecs"

query_groundtruth_extension = "_groundtruth.ivecs"
calibration_groundtruth_extension = "_calibration_groundtruth.ivecs"

def read_fvecs(filename):
    fv = np.fromfile(filename, dtype=np.float32)
    dim = fv.view(np.int32)[0]
    fv = fv.reshape(-1, dim + 1)
    return fv[:, 1:]

def read_ivecs(filename):
    iv = np.fromfile(filename, dtype=np.int32)
    dim = iv[0]
    iv = iv.reshape(-1, dim + 1)
    return iv[:, 1:]

def write_fvecs(filename, vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    n, dim = vectors.shape

    with open(filename, "wb") as f:
        for vec in vectors:
            np.array([dim], dtype=np.int32).tofile(f)
            vec.astype(np.float32).tofile(f)

def write_ivecs(filename, vectors):
    vectors = np.asarray(vectors, dtype=np.int32)
    n, dim = vectors.shape

    with open(filename, "wb") as f:
        for vec in vectors:
            np.array([dim], dtype=np.int32).tofile(f)
            vec.astype(np.int32).tofile(f)

def CalibrationData(search_database, calibrationsize = 3000):
    """
    Generated from search database.
    search database is split into search database and calibration database
    calibration dataset is used to learn threshold for ConANN
    """
    idx = np.random.choice(search_database.shape[0], calibrationsize, replace=False)
    mask = np.ones(search_database.shape[0], dtype=bool)
    mask[idx] = False
    search_database_new = search_database[mask]
    return search_database_new, search_database[idx]

def GroundTruth(query_data, search_database, k=100):
    """
    For genrating ground truth
    """
    index_gt = faiss.IndexFlatL2(search_database.shape[1])
    index_gt.add(search_database)
    # KNN search not ANN for ground truth
    print("Generating Ground Truth...")
    start_time = time.time()
    distances, indices = index_gt.search(query_data, k)
    print(f"Ground Truth Generated in: {time.time() - start_time:.2f} seconds")
    return indices

def main():
    k = 100 # ground truth with 100 NN vector indices
    xb = read_fvecs(base_path + search_dataset_extension)
    xq = read_fvecs(base_path + query_dataset_extension)
    

    #use 0.5 of size of no of query vectors
    xb ,xc = CalibrationData(xb,int(xq.shape[0]*1))

    xb = np.ascontiguousarray(xb, dtype=np.float32)
    xq = np.ascontiguousarray(xq, dtype=np.float32)
    xc = np.ascontiguousarray(xc, dtype=np.float32)
    
    # need to regenrate ground truth of query vectors due to splitting search database into search database and calibration dataset
    gt = GroundTruth(xq, xb, k)

    # Genrating ground truth for calibration dataset
    gt_xc = GroundTruth(xc, xb, k) # ground truth for calibration dataset

    write_fvecs(base_path + search_dataset_extension, xb)
    write_fvecs(base_path + calibration_dataset_extension, xc)

    write_ivecs(base_path + query_groundtruth_extension, gt)
    write_ivecs(base_path + calibration_groundtruth_extension, gt_xc)


if __name__ == "__main__":
    main()