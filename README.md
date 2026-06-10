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
Size of calibration dataset is comparable to 0.5*query dataset size for **SIFT**
and  1*query dataset size for **GIST**
```c
  python3 create_calibartion_dataset_sift.py
  python3 create_calibartion_dataset_gist.py
```
## Compilation:
```c
  gcc -O3 -march=native -ffast-math -c lib/k_means.c -o obj/k_means.o
  gcc -O3 -march=native -ffast-math -c lib/IVF_ANN.c -o obj/IVF_ANN.o -lm
  gcc -O3 -march=native -ffast-math -c lib/ConANN.c -o obj/ConANN.o -lm
```
