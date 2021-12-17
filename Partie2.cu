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
            printf("%1.1f ", M[col]);
        }
        printf("\n");
    }
}

 //Layer 2 - Convolution 2D

// __global__ void cudaConv2D(float *M, float *kernel, float *Mout, int n, int p, int kernel_size, int nb_kernel){
//     //Convolution d'une matrice par un kernel
    
//     //Calcul de la convolution
//     //définition des index de pixel dans l'image
//     int y = blockIdx.x + (kernel_size - 1)/2; //1 block = 1 ligne de l'image
//     int x = threadIdx.x + (kernel_size - 1)/2; //1 thread = 1 pixel de la ligne
    
//     int c = (kernel_size -1)/2; //centre du kernel
    
//     int idx = 0;
    

//     for (int n_k = 0; n_k<nb_kernel; n_k++){
//         float s = 0.0f;

//         if (idx < n*p*nb_kernel){
//             for (int i = 0; i<kernel_size; i++){
//                 for (int j = 0; j<kernel_size; j++){
//                    int iout = j + x - c;
//                    int jout = i + y - c;
//                 s += M[jout * n + iout] * kernel[i * kernel_size + j];
//                 }
                
//                 Mout[idx] = s;
//                 idx +=1;
                
//             }
//         }
//     }
// }


__global__ void cudaConv2D(float* M, float* kernel, float* Mout, int M_ligne, int M_colonne, int kernel_size, int nb_kernel, int Mout_ligne, int Mout_colonne)
{
	int lig = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	float s = 0.0;

	if (lig < Mout_ligne && col < Mout_colonne)
	{
		int tot = M_ligne * M_colonne;

		for (int kernel_lig = 0; kernel_lig < kernel_size; kernel_lig++) {
			for (int kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
				for (int n_k = 0; n_k < nb_kernel; n_k++)

					s += M[(lig + kernel_lig) * M_colonne + col + kernel_col + n_k * tot] * kernel[kernel_lig * kernel_size + kernel_col + n_k * nb_kernel];
			}
		}
		Mout[lig * Mout_colonne + col] = s;
	}
}



//Fonction main
int main(){
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU CPU \\\\\\\\\\\\\\\
    
    
    /////////////// INITIALISATION DES MATRICES \\\\\\\\\\\\\\\
    //Création de l'image d'entrée à convoluer
    float *raw_data;    
    raw_data = (float*)malloc(32 * 32 * 1 * sizeof(float));
    
    MatrixInit(raw_data, 32, 32, 1, 0);
    //MatrixPrint2D(raw_data, 32, 32);
    
    //Création de la sortie de la conv2D
    float *C1_data;    
    C1_data = (float*)malloc(28 * 28 * 6 * sizeof(float));
    
    MatrixInit(C1_data, 28, 28, 6, 1);
    
    //Création de la sortie du sous-échantillonnage
    float *S1_data;    
    S1_data = (float*)malloc(14 * 14 * 6 * sizeof(float));
    
    MatrixInit(S1_data, 14, 14, 6, 1);
    
    //Création des premiers noyaux de convolution
    float *C1_kernel;    
    C1_kernel = (float*)malloc(5 * 5 * 6 * sizeof(float));
    
    MatrixInit(C1_kernel, 5, 5, 6, 0);

    
    
    /////////////// TOUT CE QUI SE PASSE ICI EST FAIT DU GPU \\\\\\\\\\\\\\\
    
    //Test de cudaMatrixAdd
    float *d_raw_data, *d_C1_data, *d_C1_kernel;
    
    //Allocation des mémoires des matrices pour cuda
    cudaMalloc((void**)&d_raw_data, sizeof(float) * 32 * 32 * 1);
    cudaMalloc((void**)&d_C1_kernel, sizeof(float) * 5 * 5 * 6);
    cudaMalloc((void**)&d_C1_data, sizeof(float) * 28 * 28 * 6);

    cudaMemcpy(d_raw_data, raw_data, sizeof(float) * 32 * 32 * 1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, sizeof(float) * 5 * 5 * 6, cudaMemcpyHostToDevice);
  
    //Convolution sur GPU
    dim3 block_size(32, 32);
    dim3 grid_size(1,1);
    
    cudaConv2D<<<grid_size,block_size>>>(d_raw_data, d_C1_kernel, d_C1_data, 32, 32, 5, 6, 28, 28);
    cudaDeviceSynchronize();
    
    //Copie du résultat sur CPU
    cudaMemcpy(C1_data, d_C1_data, sizeof(float) * 28 * 28 * 6, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    
    MatrixPrint2D(C1_data, 28, 28);
    
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    
    free(raw_data);
    free(C1_data);
    free(C1_kernel);
}