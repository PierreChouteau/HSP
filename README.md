# Hardware for Signal Processing
TP : Implémentation d'un CNN - LeNet-5 sur GPU

Réalisé par : **Elisa DELHOMME && Pierre CHOUTEAU**


## 1- Objectif & Méthodes de travail

### Objectif
Les objectif de ces 4 séances de TP de HSP sont : 
* Apprendre à utiliser CUDA, Etudier la complexité de vos algorithmes et l'accélération obtenue sur GPU par rapport à une éxécution sur CPU
* Observer les limites de l'utilisation d'un GPU
* Implémenter "from scratch" un CNN : juste la partie inférence et non l'entrainement
* Exporter des données depuis un notebook python et les réimporter dans un projet cuda
* Faire un suivi de votre projet et du versionning à l'outil git

### 1.1. Implémentation d'un CNN
#### LeNet-5
L'objectif à terme de ces 4 séances est d'implémenter l'inférence dun CNN très classique : LeNet-5

La lecture de cet article vous apportera les informations nécessaires pour comprendre ce réseau de neurone.

https://www.datasciencecentral.com/profiles/blogs/lenet-5-a-classic-cnn-architecture
