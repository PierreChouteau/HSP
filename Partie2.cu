//Pierre Chouteau & Elisa Delhommé
#include <stdio.h>
#include <stdlib.h>



//Layer 1 - Génération des données de test (10 décembre 2021)

//Création d'une matrice (p lignes, n colonnes)
void MatrixInit(float *M, int n, int p, int d, int zero){
        
    float random_value;
    
    if (zero==1){
        for (int i=0; i<n*p*d; i++){
            M[i] =  0;
        }
    }
    else{
        //Valeurs entre 0 et 1
        for (int i=0; i<n*p*d; i++){
            random_value = (float)rand()/(float)(RAND_MAX/1.0);
            M[i] =  random_value;
        }
    }
}


//Affichage d'une matrice
void MatrixPrint2D(float *M, int n, int p){
        
    for (int lig=0; lig<p; lig++){
        for(int col=lig*n; col<n*(lig+1); col++){
            printf("%f ", M[col]);
        }
        printf("\n");
    }
}


//Fonction main
int main(){
    
    //Création de l'image d'entrée à convoluer
    float *raw_data;    
        
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 0);
    MatrixPrint2D(raw_data, 32, 32);
    
    //Création de la sortie de la conv2D
    float *C1_data;    
        
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 1);
    
    //Création de la sortie du sosu-échantillonnage
    float *S1_data;    
        
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 1);
    
    //Création des premiers noyaux de convolution
    float *C1_kernel;    
        
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 0);
}