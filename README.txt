Ce notebook est un pipeline d'apprentissage permettant la création et l'analyse de multiples modèles d'apprentissage appliqué à des données de croissance de plantes (W. Rymaszewski, 2017), à savoir:
	- KNN avec distance Euclidienne
	- KNN avec la distance DTW
	- Shapelet exact (A. Mueen, 2011)
	- Shapelet apprenants (J. Grabocka, 2014)

1°) Installer le gestionnaire de paquet conda :
	- https://conda.io/docs/user-guide/install/download.html

2°) Créer un environnement virtuel python 3.6 avec conda :
	- conda create -n [nom_environnement] python=3.6

3°) Activer l'environnement :
	- source activate [nom_environnement]

4°) Installer toutes les dépendances:
	- conda install pandas numpy scikit-learn keras scipy matplotlib seaborn xlrd theano jupyter
	- conda install -c conda-forge jupyter_contrib_nbextensions
	- conda install -c conda-forge python-graphviz
	- conda install -c conda-forge tslearn

5°) Lancer jupyter-notebook
	- jupyter-notebook
