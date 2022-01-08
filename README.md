# Hardware for Signal Processing
Implémentation d'un CNN - LeNet-5 sur GPU

Réalisé par : **Elisa DELHOMME & Pierre CHOUTEAU**

## Getting Started

### Prérequis

Si vous voulez continuer ce projet, ou bien utiliser une partie de celui-ci, vous n'aurez pas besoin de grand chose. 
Le langage utilisé pour ce projet est du Cuda, il vous faudra donc a minima un PC avec une carte graphique Nvidia. Sinon, vous ne pourrez pas utiliser les fonctions qui utilisent le GPU. 

#### Quel IDE pour le langage Cuda ? 
Aujourd'hui le langage Cuda n'est encore présent sur aucun IDE, mais comme la compilation et l'exécution se fait via la console, il est possible d'utiliser n'importe quel IDE. 

Un IDE comprenant la coloration synthaxque du C ou du C++ fait largement l'affaire. Choisissez donc celui qui vous fera plaisir (jupyter-lab, VsCode ou encore sublime text font largement l'affaire)


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


## 3- Partie 2. Premières couches du réseau de neurones LeNet-5 : Convolution 2D et subsampling
L'architecture du réseau LeNet-5 est composé de plusieurs couches :

* Layer 1- Couche d'entrée de taille 32x32 correspondant à la taille des images de la base de donnée MNIST
* Layer 2- Convolution avec 6 noyaux de convolution de taille 5x5. La taille résultantes est donc de 6x28x28.
* Layer 3- Sous-échantillonnage d'un facteur 2. La taille résultantes des données est donc de 6x14x14.

### 3.1. Layer 1 - Génération des données de test
### 3.2. Layer 2 - Convolution 2D
### 3.3. Layer 3 - Sous-échantillonnage
### 3.4. Tests
### 3.5. Fonctions d'activation

## 4- Partie 3. Un peu de Python

### 4.1. Notebook Python
### 4.2. Création des fonctions manquantes
### 4.3. Importation du dataset MNIST et affichage des données en console
### 4.4. Export des poids dans un fichier .h5
