//Pierre Chouteau & Elisa Delhommé
#include <stdio.h>
#include <stdlib.h>



//Partie 1 - Prise en main de Cuda (03 décembre 2021)

//Création d'une matrice (p lignes, n colonnes)
void MatrixInit(float *M, int n, int p){
        
    float random_value;
    
    //Valeurs entre -1 et 1
    for (int i = 0; i < n * p; i++){
        random_value = (float)rand() / (float)(RAND_MAX/1.0);
        M[i] =  random_value;
    }
}


//Affichage d'une matrice
void MatrixPrint(float *M, int n, int p){
        
    for (int lig = 0 ; lig < p; lig++){
        for(int col = lig * n; col < n * (lig+1); col++){
            printf("%f ", M[col]);
        }
        printf("\n");
    }
}


//Addition de deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition from the CPU...\n\n");
    
    for (int i = 0; i < n * p; i++){
        Mout[i] = M1[i] + M2[i];
    }
    
}

//Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < p && col < n){
        Mout[lig * n + col] = M1[lig * n + col] + M2[lig * n + col];
    }
}

//Multiplication de deux matrices NxN sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n){
    
    printf("Multiplication from the CPU...\n\n");
    
    for (int lig = 0; lig < n; lig++){
        for (int col = 0; col < n; col++){
            float s = 0.0f;
            for (int i = 0; i < n; i++) {
                s += M1[lig * n + i] * M2[i * n + col];
            }
            Mout[lig * n + col] = s;
        }
    }
}


//Multiplication de deux matrices NxN sur GPU
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    printf("Multiplication from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if (lig < n && col < n){
        for (int i = 0; i < n; i++){
            s += M1[lig * n + i] * M2[i * n + col];
        }
    }
    Mout[lig * n + col] = s;
}

//Fonction main
int main(){
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU CPU \\\\\\\\\\\\\\\
    
    //Test de MatrixInit et MatrixPrint
    
    float *M;    
    
    int n = 2;
    int p = 2;
    
    //Allocation de la mémoire pour la création de la matrice
    M = (float*)malloc(n * p * sizeof(float));
    
    MatrixInit(M, n, p);
    //MatrixPrint(M, n, p);
    
    //printf("\n");
    
    free(M);
    
    //Test de MatrixAdd
    float *M1;
    float *M2;
    float *Mout;
    
    //Allocation des mémoires
    M1 = (float*)malloc(n * p * sizeof(float));
    M2 = (float*)malloc(n * p * sizeof(float));
    Mout = (float*)malloc(n * p * sizeof(float));
    
    MatrixInit(M1, n, p);
    MatrixInit(M2, n, p);
    //MatrixAdd(M1, M2, Mout, n, p);
    //MatrixMult(M1, M2, Mout, n);
    
//     printf("Matrice 1\n");
//     MatrixPrint(M1, n, p);
//     printf("\nMatrice 2\n");
//     MatrixPrint(M2, n, p);
//     printf("\nMatrice résultante de la mutliplication:\n");
//     MatrixPrint(Mout, n, p);

    
    
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU GPU \\\\\\\\\\\\\\\
    
    //Test de cudaMatrixAdd
    float *d_M1, *d_M2, *d_Mout;
    
    //Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_M1, sizeof(float) * n * p);
    cudaMalloc((void**)&d_M2, sizeof(float) * n * p);
    cudaMalloc((void**)&d_Mout, sizeof(float) * n * p);

    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n * p, cudaMemcpyHostToDevice);

    //Addition sur GPU
    dim3 block_size(n, p);
    dim3 grid_size(1, 1);
    // cudaMatrixAdd<<<grid_size, block_size>>>(d_M1, d_M2, d_Mout, n, p);
    
    //Multiplication sur GPU    
    cudaMatrixMult<<<grid_size,block_size>>>(d_M1, d_M2, d_Mout, n);
    cudaDeviceSynchronize();
    
    
    //Copie du résultat sur CPU
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    printf("Matrice 1\n");
    MatrixPrint(M1, n, p);
    printf("\nMatrice 2\n");
    MatrixPrint(M2, n, p);
    printf("\nMatrice résultante de la Multiplication:\n");
    MatrixPrint(Mout, n, p);
    
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
    
    free(M1);
    free(M2);
    free(Mout);
}