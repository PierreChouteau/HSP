// Pierre Chouteau & Elisa Delhommé

#include <stdio.h>
#include <stdlib.h>



// Layer 1 - Génération des données de test (10 décembre 2021)

/*
*** Function Name : MatrixInit ***

Description : Initialiser n'importe quelle matrice de taille NxP selon diférent cas :

    * Si on veut initialiser qu'avec des 0  ==> type == 0 
    
                                    0 0 0
    * Pour avoir un kernel comme :  0 1 0   ==> type == 1
                                    0 0 0

    * Pour avoir une initisalisation aléatoire entre 0 et 1: type == 2

*/
void MatrixInit(float *M, int n, int p, int d, int type){
    
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
        M[4] = 1;
    }
    else{
        //Valeurs entre 0 et 1
        for (int i = 0; i < n * p * d; i++){
            random_value = (float)rand() / (float)(RAND_MAX/1.0);
            M[i] =  random_value;
        }
    }
}


/*
*** Function Name : MatrixPrint ***

Description : Sert à afficher n'importe quelle matrice NxP dans une forme plus conventionnelle. 

                                                              0 0 0
ex : M = [0 0 0; 0 0 0; 0 0 0] sera affichée comme suit : M = 0 0 0   
                                                              0 0 0 
*/
void MatrixPrint2D(float *M, int n, int p){
        
    for (int lig = 0; lig < p; lig++){
        for(int col = lig * n; col < n * (lig+1); col++){
            printf("%1.1f ", M[col]);
        }
        printf("\n");
    }
}


// Layer 2 - Convolution 2D

__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne){
    
    // Convolution d'une matrice par un kernel
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;

    if (lig < Mout_ligne && col < Mout_colonne){
        int tot = M_ligne * M_colonne;

        for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
            for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                for (int n_k = 0; n_k < nb_kernel; n_k++){
                    s += M[(lig + kernel_lig) * M_colonne + col + kernel_col + n_k * tot] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
            
                }
            }
        }
        Mout[lig * Mout_colonne + col] = s;
    }
}


// Layer 3 - Sous-échantillonnage 

__global__ void cudaMeanPool(float* M, float* Mout, int M_ligne, int M_colonne, int profondeur, int meanpool_size, int Mout_ligne, int Mout_colonne){
    
    // MeanPool d'une matrice par un kernel 2x2
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float s = 0.0;
    int tot_meanpool = meanpool_size * meanpool_size;

    if (lig % meanpool_size == 0 && col % meanpool_size == 0){
        int tot = M_ligne * M_colonne;

        for (int meanpool_lig = 0; meanpool_lig < meanpool_size; meanpool_lig++) {
            for (int meanpool_col = 0; meanpool_col < meanpool_size; meanpool_col++) {
                for (int n_prof = 0; n_prof < profondeur; n_prof++){
                    s += M[(lig + meanpool_lig) * M_colonne + col + meanpool_col + n_prof * tot] / tot_meanpool;
            
                }
            }
        }
        if (lig == 0){
            Mout[lig * Mout_colonne + (col / meanpool_size)] = s;
    
        }
        else if (col == 0){
            Mout[(lig / meanpool_size) * Mout_colonne + col] = s;
    
        }
        else{
            Mout[(lig / meanpool_size) * Mout_colonne + (col / meanpool_size)] = s;
        }
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



// Layer 4 - Dense | Linear

/*
*** Function Name : cudaMatrixMultGeneral ***

Description : Sert à effectuer la multiplication matricielle (dot) d'une matrice NxP avec une matrice PxM sur le GPU

*/
__device__ float* cudaMatrixMultGeneral(float *M1, float *M2, float *Mout, int n, int p, int m){
    
    printf("Multiplication from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if (lig < n && col < m){
        for (int i = 0; i < p; i++){
            s += M1[lig * p + i] * M2[i * m + col];
        }
        Mout[lig * m + col] = s;
    }
    
    return Mout;
}


/*
*** Function Name : MatrixAdd ***

Description : Sert à effectuer la multiplication matricielle (dot) de deux matrices carrées NxN sur GPU

*/
__device__ float* cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < n && col < p){
        Mout[lig * p + col] = M1[lig * p + col] + M2[lig * p + col];
    }
    
    return Mout;
}


__global__ void cudaDense(float* d_M, float* d_Mout, float* d_W, float* d_b, int n, int p, int m){
    
    d_Mout = cudaMatrixMultGeneral(d_M, d_W, d_Mout, n, p, m);
    d_Mout = cudaMatrixAdd(d_Mout, d_b, d_Mout, n, p);
    
}

// Fonction main
int main(){
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU CPU \\\\\\\\\\\\\\\    
    //**********************************************************************
    
    // INITIALISATION DES MATRICES pour le CPU \\
    
    // Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 2);
    
    // Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 0);
    
    // Création de la sortie de la conv2D
    float *C2_data;    
    C2_data = (float*)malloc(10 * 10 * 6 * sizeof(float));
    
    MatrixInit(C2_data, 10, 10, 6, 0);
    
    // Création de la sortie du sous-échantillonnage
    float *S2_data;    
    S2_data = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 5, 5, 6, 0);
    
    // Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 1);

    
    // INITIALISATION DES MATRICES pour le GPU \\
    
    // Définition des matrices cuda
    float *d_raw_data, *d_C1_data, *d_C1_kernel, *d_S1_data, *d_C2_data, *d_S2_data;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);
    cudaMalloc((void**)&d_S1_data, sizeof(float) * 14 * 14 * 6);
    cudaMalloc((void**)&d_C2_data, sizeof(float) * 10 * 10 * 6);
    cudaMalloc((void**)&d_S2_data, sizeof(float) * 5 * 5 * 6);
    
    // Copie des valeurs des matrices initialisées sur le CPU dans leur homonyme GPU
    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_data, C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S1_data, S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C2_data, C2_data, sizeof(float) * 10 * 10 * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_S2_data, S2_data, sizeof(float) * 5 * 5 * 16, cudaMemcpyHostToDevice);
  
    
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU CPU \\\\\\\\\\\\\\\
    //**********************************************************************
    
    // Process sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);
    
    cudaConv2D<<<grid_size,block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C1_data, 28*28);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C1_data, d_S1_data, 28, 28, 6, 2, 14, 14);
    cudaDeviceSynchronize();
    
    cudaConv2D<<<grid_size,block_size>>>(d_S1_data, d_C1_kernel, d_C2_data, 14, 14, 5, 16, 10, 10);
    cudaDeviceSynchronize();
    
    cudaTanh<<<grid_size, block_size>>>(d_C2_data, 10*10);
    cudaDeviceSynchronize();
    
    cudaMeanPool<<<grid_size, block_size>>>(d_C2_data, d_S2_data, 10, 10, 16, 2, 5, 5);
    cudaDeviceSynchronize();
    
    
    
    // Copie des résultats sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S1_data, d_S1_data, sizeof(float) * 14 * 14 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(C2_data, d_C2_data, sizeof(float) * 10 * 10 * 6, cudaMemcpyDeviceToHost);
    cudaMemcpy(S2_data, d_S2_data, sizeof(float) * 5 * 5 * 6, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Affichage de la matrice résultat
    MatrixPrint2D(S2_data, 5, 5);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C2_data);
    cudaFree(d_S2_data);
    
    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(C2_data);
    free(S2_data);
}