# Hardware for Signal Processing
Implémentation d'un CNN - LeNet-5 sur GPU

Réalisé par : **Elisa DELHOMME & Pierre CHOUTEAU**

## Getting Started

### Prérequis

Si vous voulez continuer ce projet, ou bien utiliser une partie de celui-ci, vous n'aurez pas besoin de grand chose. 
Le langage utilisé pour ce projet est du Cuda, il vous faudra donc a minima un PC avec une carte graphique Nvidia. Sinon, vous ne pourrez pas utiliser les fonctions qui utilisent le GPU. 

#### Quel IDE pour le langage Cuda ? 
Aujourd'hui le langage Cuda n'est encore présent sur aucun IDE, mais comme la compilation et l'exécution se font via la console, il est possible d'utiliser n'importe quel IDE. 

Un IDE comprenant la coloration synthaxique du C ou du C++ fait largement l'affaire. Choisissez donc celui qui vous voulez (Jupyter-Lab, VsCode ou encore Sublime Text font largement l'affaire)


#### Compilation et Execution depuis la console

Pour compiler un code Cuda, il vous suffit de lancer la commande : 

```
nvcc nomdufichier.cu -o nomdufichier
```

Quand vous aurez fait cela, vous verrez apparaître un fichier portant le nom "nomdufichier". 
Vous n'avez donc plus qu'à l'exécuter, et là encore, rien de plus simple. Lancer simplement la commande : 

```
./nomdufichier
```

PS: Pour que ces commandes fonctionnent il faut bien sûr que vous soyez dans votre dossier de travail. Vous pouvez vous déplacer facilement dans les dossier grâce à la commande "cd".


## 1- Objectif
Les objectif de ce projet sont : 
* Apprendre à utiliser CUDA
* Etudier la complexité d'algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda
* Faire un suivi de votre projet et du versionning grâce à l'outil git

### Implémentation d'un CNN
#### LeNet-5
A terme, l'objectif final est d'implémenter l'inférence dun CNN très classique : LeNet-5

La lecture de cet article apporte les informations nécessaires pour comprendre ce réseau de neurones.

https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture

<h1 align=left><img src="LeNet-5.png"></h1>


## 2- Partie 1. Prise en main de Cuda : Addition et Multiplication de matrices

### 2.1. Création de matrice
Dans tout le projet, que ce soit sur CPU ou GPU, on souhaite représenter les matrices sous forme de listes constituées des lignes de la matrice.
Dans la fonction d'initialisation, on initialise la matrice **N*P** avec des valeurs aléatoires.
![image](https://user-images.githubusercontent.com/94063629/148687288-f1d8b3a1-a6b9-4ab7-af0a-11b4140eb562.png)

### 2.2. Affichage de matrice sous forme conventionnelle
La création d'une fonction d’affichage de matrice sous sa forme classique, avec chaque ligne affichée l’une en dessous l’autre, est utile, voire même nécessaire afin de pouvoir vérifier le bon fonctionnement des opérations traitées ou prendre connaissance d'un résultat.
![image](https://user-images.githubusercontent.com/94063629/148687259-6fef5698-6d92-4c96-a2bf-ea60d089cf7e.png)

### 2.3. Addition
#### **CPU**
Sur **CPU**, on additionne deux matrices simplement comme à notre habitude en sommant les coefficients de chacune deux à deux puisque la représentation sous forme de liste n'est pas un frein à cette addition classique.
![image](https://user-images.githubusercontent.com/94063629/148687251-c18e9d34-435b-428f-af89-cd3c93302b13.png)

#### **GPU**
Sur **GPU**, le calcul se base également sur la somme des coefficients deux à deux mais est un peu plus complexe à mettre en oeuvre en raison de la parallélisation des calculs sur GPU.
En particulier, la définition des indices des coefficients matriciels se fait cette fois via les variables _dim3_ définissant les threads afin de paralléliser (et donc accélérer) les calculs).
![image](https://user-images.githubusercontent.com/94063629/148687472-8454bbe7-95fc-4721-b2ce-a9f924f99767.png)

On définira par exemple l'indice de la ligne à considérer (au début de la fonction, et qui changera à chaque itération) par:
```
int lig = blockIdx.y * blockDim.y + threadIdx.y
```
où **_blockIdx.y_** désigne le numéro de la ligne du _block_ dans la _grid_,

   **_blockDim.y_** donne le nombre de lignes total de ce _block_,
   
   **_threadIdx.y_** donne le numéro de la ligne du _thread_ appartenant au _block_.

**Ce qui suit est nécessaire pour tout calcul effectué sur le GPU, et est donc également nécessaire pour la réalisation de la multiplication.**

Il est, en outre, nécessaire de définir la fonction avec __global__ afin d’effectuer les calculs sur le GPU en l’appelant depuis le CPU, par exemple dans le cas de l'addition de matrices sur GPU:
```
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p);
```
Ce sera le cas pour toutes les fonctions effectuant des calculs sur le GPU tout en étant appelé depuis le CPU.

Les mémoires des matrices doivent être allouées sur le GPU avec ```cudaMalloc``` et copiées depuis le CPU vers le GPU avec ```cudaMemcpy```
Pour que la parallélisation soit fonctionnelle et efficace il est nécessaire de définir **en dehors de la fonction** (dans le main par exemple) les dimensions des variables _dim3_.
```
dim3 block_size(n, m);
dim3 grid_size(1, 1);
```
où _**n**_ et _**m**_ sont les dimensions de la matrice résultante du calcul.

Enfin, la fonction d'addition sur le GPU est appelée via la commande suivante:
```
cudaMatrixAdd<<<grid_size, block_size>>>(d_M1, d_M2, d_Mout, n, p);
```

### 2.4. Multiplication
#### **CPU**
La multiplication de deux matrices sur le CPU se fait de façon habituelle. La seule difficulté réside dans l'indexage correct des coefficients recherchés, les matrices étant sous forme de liste.
![image](https://user-images.githubusercontent.com/94063629/148688381-221ddec3-26b4-46ba-b3d4-df48913f3031.png)

#### **GPU**
Comme pour l'addition, la multiplication sur GPU repose sur le même principe que la multiplication classique mais les indexes lignes et colonnes désirées doivent être définies (exactement selon la même formulation que pour l'addition) avec les variables définissant les threads et blocks.

## 3- Partie 2. Premières couches du réseau de neurones LeNet-5 : Convolution 2D et subsampling
L'architecture du réseau LeNet-5 est composé de plusieurs couches :

* Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de données MNIST
* Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultante est donc de 6x28x28.
* Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultante des données est donc de 6x14x14.

### 3.1. Layer 1 - Génération des données de test
La génération des données consiste en la création des matrices suivantes sous la forme de tableaux à une dimension:
- la matrice d'entrée dans le réseau 32x32 **raw_data** initialisée avec des valeurs aléatoires entre 0 et 1.
- la matrice 6x28x28 **C1_data** résultante de la convolution 2D initialisée avec des valeurs nulles.
- la matrice 6x14x14 **S1_data** résultante du sous-échantillonnage initialisée avec des valeurs nulles.
- le kernel 6x5x5 **C1_kernel** permettant la convolution de la layer 1 et initialisé entre 0 et 1.

### 3.2. Layer 2 - Convolution 2D
La convolution se fait exclusivement sur le GPU. De façon analogue à la multiplication, on fait glisser un kernel **C1_kernel** sur la totalité de la matrice **raw_data** pour obtenir la matrice résultante **C1_data**.

![Peek 2022-01-10 00-30](https://user-images.githubusercontent.com/75682374/148705636-8c98babd-8159-4ac1-b08b-8abeef920215.gif)


### 3.3. Layer 3 - Sous-échantillonnage
Le sous-échantillonage se fait par une fonction de _MeanPooling_, à savoir un moyennage sur une fenêtre glissante 2x2 (afin de réduire par 2 les dimensions de **raw_data** et d'obtenir **S1_data**.
![image](https://user-images.githubusercontent.com/94063629/148700559-b2a5e8f8-9c98-4717-8b42-48f23100e9c8.png)
Celui-ci se fait également sur le GPU depuis un appel du CPU.

### 3.4. Tests
### 3.5. Fonctions d'activation
Dans l'objectif de parfaire le réseau de neurones, une couche d'activation est requise. Comme on peut le remarquer dans l'article sur l'architecture de LeNet-5, la fonction d'activation utilisée est une tangente hyperbolique (_tanh_). Celle-ci interviendra après chaque layer de _Conv2D_.
Afin de se laisser la possibilité d'appeler cette fonction d'activation depuis chaque kernel sur le GPU, on définit cette fois la fonction avec le _specifier_ ```__device__```, et non ```__global__``` pour effectuer les calculs sur le GPU depuis un appel du GPU.

## 4- Partie 3. Un peu de Python

### 4.1. Notebook Python
Dans cette dernière partie, on utilise le notebook Python comme référence afin de finaliser notre réseau LeNet5.
En particulier, celui-ci nous servira, grâce à un entraînement rapide, d'obtenir les valeurs optimales des poids de chaque couche afin de pouvoir initialiser les _kernels_ de convolution et les poids des couches _fully connected_ de façon à obtenir les meilleurs résultats.

En effet, dans ce projet, on ne désire pas créer la fonction d'entrainement du réseau de neurones en Cuda, car cela est beaucoup plus complexe, et nous aurait pris trop de temps à mettre en place (*Descente de gradient, BackPropagation...*). Ceci pourrait donc être un bon point de départ pour continuer ce projet :
* Créer l'optimizer
* La fonction de loss
* L'entrainement sur une base de test et de validation.

### 4.2. Création des fonctions manquantes
On construit le réseau en ajoutant couches de convolution et de _MeanPooling_. Il est également nécessaire de créer une couche de _Dense_ effectuant l'opération **W.x + b** où W sont les poids et b, les biais appliqués à l'image d'entrée x.
Cette fonction fait intervenir les fonctions de multiplication et d'addition sur le GPU.

### 4.3. Importation du dataset MNIST et affichage des données en console
### 4.4. Export des poids dans un fichier .h5
Après avoir entraîné le réseau dans le notebook et récupéré les poids et biais de chaque couche, on les utilise pour initialiser les _kernels_. 

Le réseau LeNet5 est désormais entièrement fonctionnel pour une image d'entrée de dimensions 32x32.
