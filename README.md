# Hardware for Signal Processing
TP : Implémentation d'un CNN - LeNet-5 sur GPU

Réalisé par : **Elisa DELHOMME & Pierre CHOUTEAU**

## Getting Started

### Prerequisites 

### Installing


## 1- Objectif
Les objectif de ce TP sont : 
* Apprendre à utiliser CUDA, Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda
* Faire un suivi de votre projet et du versionning grâce à l'outil git

### Implémentation d'un CNN
#### LeNet-5
A terme, l'objectif final est d'implémenter l'inférence dun CNN très classique : LeNet-5

La lecture de cet article vous apportera les informations nécessaires pour comprendre ce réseau de neurone.

https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture

<h1 align=left><img src="LeNet-5.png"></h1>


## 2- Partie 1. Prise en main de Cuda : Addition et Multiplication de matrices


## 3- Partie 2. Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling
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