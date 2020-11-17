#%% package
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

#%% def unsupervisedTraining
def unsupervisedTraining(featLearn, method='kmeans'):
    ''' apprentissage avec la fonction KMeans() et GaussianMixture 
    de scikit-learn :
    - featLearn est la matrice de l'ensemble d'apprentissage
    - method: type d'algorithme de Machine Learning utilisé (KMeans et GaussianMixture) 
    - nbCluster est le nombre de cluster = nombre de classes rentré par l'utilisateur en début de fonction
    
    - renvoie model: le modèle de classement ou classifieur
    '''
    
    
    # fixer le nombre de classes
    answer = input('nombre de classes:')
    nbCluster = int(answer)
    
    if method == 'kmeans':

    # ------------------------------------------------
    # YOUR CODE HERE








    # ------------------------------------------------
    # ------------------------------------------------


    elif method == 'gmm':
    
    # ------------------------------------------------
    # YOUR CODE HERE
   





    # ------------------------------------------------
    # ------------------------------------------------


    # sortie
    return model



