//Pierre Chouteau & Elisa Delhommé
#include <stdio.h>
#include <stdlib.h>



//Partie 1 - Prise en main de Cuda (03 décembre 2021)

//Création d'une matrice (p lignes, n colonnes)
void MatrixInit(float *M, int n, int p){
    
    float random_value;
    
    //Valeurs entre -1 et 1
    for(int i=0; i<n*p; i++){
        random_value = (float)rand()/(float)(RAND_MAX/1.0);
        M[i] =  random_value;
    }
}


//Affichage d'une matrice
void MatrixPrint(float *M, int n, int p){
    
    for(int i=0; i<p; i++){
        for(int j=i*n; j<n*(i+1); j++){
            printf("%f ", M[j]);
        }
        printf("\n");
    }
}


//Addition de deux matrices sur CPU
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    for(int i=0; i<n*p;i++){
        Mout[i] = M1[i]+M2[i];
    }
    
}

//Addition de deux matrices sur GPU
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
}

//Multiplication de deux matrices NxN sur CPU
void MatrixMult(float *M1, float *M2, float *Mout, int n){
    
}


//Multiplication de deux matrices NxN sur GPU
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    
}

//Fonction main
int main(){
    
    printf("Hello from the CPU!\n\n");
    
    //Test de MatrixInit et MatrixPrint
    printf("Here is our matrix!\n\n");
    
    float *M;    
    
    int n = 3;
    int p = 3;
    
    //Allocation de la mémoire pour la création de la matrice
    M = (float*)malloc(n * p * sizeof(float));
    
    MatrixInit(M, n, p);
    MatrixPrint(M, n, p);
    
    printf("\n");
    
    
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
    MatrixAdd(M1, M2, Mout, n, p);
    
    printf("Matrice 1\n");
    MatrixPrint(M1, n, p);
    printf("\nMatrice 2\n");
    MatrixPrint(M2, n, p);
    printf("\nMatrice résultante de la somme:\n");
    MatrixPrint(Mout, n, p);
}