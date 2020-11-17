#%% Machine Learning Class - Exercise Aral Sea Surface Estimation

# package
import numpy as np
import matplotlib.pyplot as plt

plt.ioff() # to see figure avant input


# ------------------------------------------------
# YOUR CODE HERE
from preprocessing import preprocessing






# ------------------------------------------------
# ------------------------------------------------





#%% Examen des données, prétraitements et extraction des descripteurs

# Chargement des données et prétraitements
featLearn,img73,img87 = preprocessing()




#%% Apprentissage / Learning / Training


# Apprentissage de la fonction de classement

# ------------------------------------------------
# YOUR CODE HERE



# ------------------------------------------------
# ------------------------------------------------




# prediction des labels sur la base d'apprentissage

# ------------------------------------------------
# YOUR CODE HERE


# ------------------------------------------------
# ------------------------------------------------




# Visualisation des resultats

# ------------------------------------------------
# YOUR CODE HERE






# ------------------------------------------------
# ------------------------------------------------



#%% Classement et estimation de la diminution de surface
# Classifying / Predicting / Testing


# mise en forme de l'image de 1973 et 1987 en matrice Num Pixels / Val Pixels

# ------------------------------------------------
# YOUR CODE HERE















# ------------------------------------------------
# ------------------------------------------------


# Classement des deux jeux de données et visualisation des résultats en image

# ------------------------------------------------
# YOUR CODE HERE


























# ------------------------------------------------
# ------------------------------------------------



#%% Estimation de la surface perdue
answer = input('Numero de la classe de la mer ? ')
cl_mer = int(answer)



# ------------------------------------------------
# YOUR CODE HERE










# ------------------------------------------------
# ------------------------------------------------



