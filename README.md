# Fuzzy_Kmeans_ConANN
implementation of ConANN with fuzzy Kmeans clustering
## Compilation:
```c
  gcc -O3 -march=native -ffast-math -c lib/k_means.c -o obj/k_means.o
  gcc -O3 -march=native -ffast-math -c lib/IVF_ANN.c -o obj/IVF_ANN.o -lm
  gcc -O3 -march=native -ffast-math -c lib/ConANN.c -o obj/ConANN.o -lm
```
