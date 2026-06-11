# Fuzzy_Kmeans_ConANN
## Implementation of ConANN with fuzzy Kmeans clustering
## Author:
  1. Suman Polley
  2. Madhumita Das
  3. Raj Lohar
  4. Bhagavath Chukka
  5. Tanish Kothari


## Downloading and extracting Data
**SIFT1M** dataset
```c
  wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
  tar -xvzf sift.tar.gz
```
**GIST1M** dataset
```c
  wget ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
  tar -xvzf gist.tar.gz
```
## Creating Calibration Dataset
Instaling necessary libraries after creating an virtual environment to not corrupt global python installations
**Ubuntu22.04**
```c
  sudo apt update
  sudo apt install python3-venv
  python -m venv myenv
  source myenv/bin/activate
  python -m pip install --upgrade pip
  pip install numpy faiss-cpu
```
**Calibration Dataset** is created from Search Database by random sampling.
Size of calibration dataset is comparable to 0.5 $\times$ query dataset size for **SIFT**
and  1$\times$ query dataset size for **GIST**
```c
  python3 create_calibartion_dataset_sift.py
  python3 create_calibartion_dataset_gist.py
```
## Compilation:
###  Library files
```c
  gcc -O3 -march=native -ffast-math -c lib/k_means.c -o obj/k_means.o
  gcc -O3 -march=native -ffast-math -c lib/IVF_ANN.c -o obj/IVF_ANN.o -lm
  gcc -O3 -march=native -ffast-math -c lib/ConANN.c -o obj/ConANN.o -lm
```
###  Experiments
####  IVF ANN
standard
```c
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o experiments/ivf_ann_sift_1m.c -Ilib -o bin/ivf_ann_sift -lm
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o experiments/ivf_ann_gist_1m.c -Ilib -o bin/ivf_ann_gist -lm
```
(Our implementation) with fuzzy clustering
```c
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o experiments/ivf_ann_sift_1m_fuzzy_clustering.c -Ilib -o bin/ivf_ann_sift_fuzzy -lm
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o experiments/ivf_ann_gist_1m_fuzzy_clustering.c -Ilib -o bin/ivf_ann_gist_fuzzy -lm
```
#### ConANN
standard
```c
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o obj/ConANN.o experiments/conann_sift_1m.c -Ilib -o bin/conann_sift -lm
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o obj/ConANN.o experiments/conann_gist_1m.c -Ilib -o bin/conann_gist -lm
```
(Our implementation) with fuzzy clustering
```c
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o obj/ConANN.o experiments/conann_sift_1m_fuzzy_clustering.c -Ilib -o bin/conann_sift_fuzzy -lm
  gcc -O3 -march=native -ffast-math -fopenmp obj/k_means.o obj/IVF_ANN.o obj/ConANN.o experiments/conann_gist_1m_fuzzy_clustering.c -Ilib -o bin/conann_gist_fuzzy -lm
```
##  Run:
####  IVF ANN
standard
```c
  bin/ivf_ann_sift
  bin/ivf_ann_gist
```
(Our implementation) with fuzzy clustering
```c
  bin/ivf_ann_sift_fuzzy
  bin/ivf_ann_gist_fuzzy
```
#### ConANN
standard
```c
  bin/conann_sift
  bin/conann_gist
```
(Our implementation) with fuzzy clustering
```c
  bin/conann_sift_fuzzy
  bin/conann_gist_fuzzy
```
