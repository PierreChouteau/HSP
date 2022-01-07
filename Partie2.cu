//Pierre Chouteau & Elisa Delhommé
#include <stdio.h>
#include <stdlib.h>



//Layer 1 - Génération des données de test (10 décembre 2021)

//Création d'une matrice (p lignes, n colonnes)
void MatrixInit(float *M, int n, int p, int d, int type){

    //Si on veut initialiser qu'avec des 0, type == 0 
    // Pour avoir un kernel comme suit, type == 1: 
    // 0, 0, 0
    // 0, 1, 0
    // 0, 0, 0
    // Pour avoir une initisalisation aléatoire entre 0 et 1, type == 2
    
    float random_value;
    
    if (type == 0){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
    }
    else if (type == 1){
        for (int i = 0; i < n * p * d; i++){
            M[i] =  0;
        }
        M[4] == 1;
    }
    else{
        //Valeurs entre 0 et 1
        for (int i = 0; i < n * p * d; i++){
            random_value = (float)rand() / (float)(RAND_MAX/1.0);
            M[i] =  random_value;
        }
    }
}


//Affichage d'une matrice
void MatrixPrint2D(float *M, int n, int p){
        
    for (int lig = 0; lig < p; lig++){
        for(int col = lig * n; col < n * (lig+1); col++){
            printf("%1.1f ", M[col]);
        }
        printf("\n");
    }
}


//Layer 2 - Convolution 2D

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne){
    
    //Convolution d'une matrice par un kernel
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;

    if (lig < Mout_ligne && col < Mout_colonne)
    {
        int tot = M_ligne * M_colonne;

        for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                for (int n_k = 0; n_k < nb_kernel; n_k++)
                {
                    s += M[(lig + kernel_lig) * M_colonne + col + kernel_col + n_k * tot] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
            
                }
            }
        }
        Mout[lig * Mout_colonne + col] = s;
    }
}


// Layer 3 - Sous-échantillonnage 

__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int profondeur, int meanpool_size, int Mout_ligne, int Mout_colonne){
    
    //MeanPool d'une matrice par un kernel 2x2
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;
    int tot_meanpool = meanpool_size * meanpool_size;

    if (lig < Mout_ligne && col < Mout_colonne)
    {
        int tot = M_ligne * M_colonne;

        for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
            for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                for (int n_prof = 0; n_prof < profondeur; n_prof++)
                {
                    s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot] / tot_meanpool;
            
                }
            }
        }
        Mout[lig * Mout_colonne + col] = s;
    }
}


// Fonction d'activation - Tanh 

__device__ float* activation_tanh(float* M, int nThreads){
    
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nThreads; i+= blockDim.x * gridDim.x){
        M[i] = tanh(M[i]);
    }
    
    return M;
}


__global__ void cudaTanh(float* M, int nThreads){
    activation_tanh(M, nThreads);
}


//Fonction main
int main(){
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU CPU \\\\\\\\\\\\\\\
    
    
    /////////////// INITIALISATION DES MATRICES \\\\\\\\\\\\\\\
    //Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 2);
    
    //Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    //Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    
    //Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 2);

    
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU GPU \\\\\\\\\\\\\\\
    
    //Test de cudaMatrixAdd
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data;
    
    //Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);

    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
  
    //Convolution sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);
    
    cudaConv2D<<<grid_size,block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C1_data, 28*28);
    cudaDeviceSynchronize();
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();
    
    
    
    //Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MatrixPrint2D(C1_data, 28, 28);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
}