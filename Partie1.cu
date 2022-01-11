// Pierre Chouteau & Elisa Delhommé

#include <stdio.h>
#include <stdlib.h>



// Partie 1 - Prise en main de Cuda (03 décembre 2021)

/*
*** Function Name : MatrixInit ***

Sert à initialiser n'importe quelle matrice de taille NxP avec des valeurs entre -1 et 1 

Paramètres : 
    n : nombre de lignes de la matrice,
    p : nombre de colonnes de la matrice si n différent de p,
    M : pointeur de la matrice
*/
void MatrixInit(float *M, int n, int p){
        
    float random_value;
    
    // Valeurs entre -1 et 1
    for (int i = 0; i < n * p; i++){
        random_value = (float)rand() / (float)(RAND_MAX/1.0);
        M[i] =  random_value;
    }
}


/*
*** Function Name : MatrixPrint ***

Sert à afficher une matrice NxP dans une forme plus conventionnelle. 

                                                              0 0 0
ex : M = [0 0 0; 0 0 0; 0 0 0] sera affichée comme suit : M = 0 0 0   
                                                              0 0 0 

Paramètres : 
    n : nombre de lignes de la matrice,
    p : nombre de colonnes de la matrice si n différent de p,
    M : pointeur de la matrice
*/
void MatrixPrint(float *M, int n, int p){
        
    for (int lig = 0 ; lig < n; lig++){
        for(int col = lig * p; col < p * (lig+1); col++){
            printf("%f ", M[col]);
        }
        printf("\n");
    }
}


/*
*** Function Name : MatrixAdd ***

Sert à additionner deux matrices de même taille NxP sur le CPU

Paramètres : 
    n : nombre de lignes des matrice,
    p : nombre de colonnes des matrice si n différent de p,
    M1 : pointeur de la matrice 1 de taille NxP,
    M2 : pointeur de la matrice 2 de taille NxP,
    Mout : pointeur vers la matrice résultante de l'addition de taille NxP
*/
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition from the CPU...\n\n");
    
    for (int i = 0; i < n * p; i++){
        Mout[i] = M1[i] + M2[i];
    }
    
}

/*
*** Function Name : cudaMatrixAdd ***

Sert à additionner deux matrices de même taille NxP sur le GPU 

Paramètres : 
    n : nombre de lignes des matrice,
    p : nombre de colonnes des matrices si n différent de p,
    M1 : pointeur de la matrice 1 de taille NxP,
    M2 : pointeur de la matrice 2 de taille NxP,
    Mout : pointeur vers la matrice résultante de l'addition de taille NxP,
    
On peut considérer les dimensions des matrices comme les paramètres gridDim et blockDim pour l'appel de la fonction:
    les lignes correspondent aux blocks,
    les colonnes correspondent aux threads
*/
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    printf("Addition from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (lig < n && col < p){
        Mout[lig * p + col] = M1[lig * p + col] + M2[lig * p + col];
    }
}

/*
*** Function Name : MatrixMult***

Sert à effectuer la multiplication matricielle (dot) de deux matrices carrées NxN sur CPU

Paramètres : 
    n : nombre de lignes et colonne des matrices,
    M1 : pointeur de la matrice 1 de taille NxN,
    M2 : pointeur de la matrice 2 de taille NxN,
    Mout : pointeur vers la matrice résultante de l'addition de taille NxN,
*/
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


/*
*** Function Name : cudaMatrixMult ***

Sert à effectuer la multiplication matricielle (dot) de deux matrices carrées NxN sur GPU

Paramètres : 
    n : nombre de lignes et de colonnes des matrices,
    M1 : pointeur de la matrice 1 de taille NxN,
    M2 : pointeur de la matrice 2 de taille NxN,
    Mout : pointeur vers la matrice résultante de la multiplication de taille NxN,
    
On peut considérer les dimensions des matrices comme les paramètres gridDim et blockDim pour l'appel de la fonction:
    les lignes correspondent aux blocks,
    les colonnes correspondent aux threads
*/
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    printf("Multiplication from the GPU...\n\n");
    
    int lig = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float s = 0.0f;
    
    if (lig < n && col < n){
        for (int i = 0; i < n; i++){
            s += M1[lig * n + i] * M2[i * n + col];
        }
        Mout[lig * n + col] = s;
    }
}


/*
*** Function Name : cudaMatrixMultGeneral ***

Sert à effectuer la multiplication matricielle (dot) d'une matrice NxP avec une matrice PxM sur le GPU

Paramètres : 
    n : nombre de lignes de la matrice M1
    p : nombre de colonnes de M1, de lignes de M2
    m : nombre de colonnes de M2
    M1 : pointeur de la matrice 1 de taille NxP,
    M2 : pointeur de la matrice 2 de taille PxM,
    Mout : pointeur vers la matrice résultante de la multiplication de taille NxM

On peut considérer les dimensions de la matrice de sortie comme les paramètres gridDim et blockDim pour l'appel de la fonction:
    les lignes correspondent aux blocks : n
    les colonnes correspondent aux threads : m
*/
__global__ void cudaMatrixMultGeneral(float *M1, float *M2, float *Mout, int n, int p, int m){
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
}

// Fonction main
int main(){
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU CPU \\\\\\\\\\\\\\\
    //**********************************************************************
    
    // Test de MatrixInit et MatrixPrint
    float *M;
    
    int n = 3;
    int p = 2;
    int m = 3;
    
    // Allocation de la mémoire pour la création de la matrice
    M = (float*)malloc(n * p * sizeof(float));
    
    MatrixInit(M, n, p);
//    MatrixPrint(M, n, p);
//    printf("\n");
    
    free(M);
    
    // Test de MatrixAdd
    float *M1;
    float *M2;
    float *Mout;
    
    // Allocation des mémoires
    M1 = (float*)malloc(n * p * sizeof(float));
    M2 = (float*)malloc(p * m * sizeof(float));
    Mout = (float*)malloc(n * m * sizeof(float));
    
    MatrixInit(M1, n, p);
    MatrixInit(M2, p, m);
    
    // Test de MatrixAdd et MatrixMult sur CPU
//    MatrixAdd(M1, M2, Mout, n, p);
//    MatrixMult(M1, M2, Mout, n);
    
//     printf("Matrice 1\n");
//     MatrixPrint(M1, n, p);
//     printf("\nMatrice 2\n");
//     MatrixPrint(M2, n, p);
//     printf("\nMatrice résultante de la mutliplication:\n");
//     MatrixPrint(Mout, n, p);

    
    
    // Test de cudaMatrixAdd
    float *d_M1, *d_M2, *d_Mout;
    
    // Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_M1, sizeof(float) * n * p);
    cudaMalloc((void**)&d_M2, sizeof(float) * p * m);
    cudaMalloc((void**)&d_Mout, sizeof(float) * n * m);

    cudaMemcpy(d_M1, M1, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * p * m, cudaMemcpyHostToDevice);
    
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU GPU \\\\\\\\\\\\\\\
    //**********************************************************************
    
    dim3 block_size(n, m);
    dim3 grid_size(1, 1);
    
    // Addition sur GPU
//    cudaMatrixAdd<<<grid_size, block_size>>>(d_M1, d_M2, d_Mout, n, p);
    
    // Multiplication sur GPU    
    cudaMatrixMultGeneral<<<grid_size,block_size>>>(d_M1, d_M2, d_Mout, n, p, m);
    cudaDeviceSynchronize();
    
    
    // Copie du résultat sur CPU
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    // Affichage du résultat de la multiplication
    printf("Matrice 1\n");
    MatrixPrint(M1, n, p);
    printf("\nMatrice 2\n");
    MatrixPrint(M2, p, m);
    printf("\nMatrice résultante de la Multiplication:\n");
    MatrixPrint(Mout, n, m);
    
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);
    
    free(M1);
    free(M2);
    free(Mout);
}