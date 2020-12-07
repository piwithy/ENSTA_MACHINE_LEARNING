import Prepocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from pythonTools import plot_batch


import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    DATASET_PATH = 'gdrive/My Drive/Colab Notebooks/ex06_supervised_seabedClassification/dataset/imgs/'
    LABEL_PATH = 'gdrive/My Drive/Colab Notebooks/ex06_supervised_seabedClassification/dataset/labels/labels.csv'
else:
    IN_COLAB = False
    DATASET_PATH = r'./dataset/imgs/'
    LABEL_PATH = r'./dataset/labels/labels.csv'

# Charger le fichier CSV
dataset_df = pd.read_csv(LABEL_PATH)
dataset_df['image_path'] = dataset_df.apply(lambda row: (DATASET_PATH + row["id"]), axis=1)


# Charger les images et les labels
images = np.array([plt.imread(img) for img in dataset_df['image_path'].values.tolist()])
label_names = dataset_df['seafloor'].to_numpy()

# transformer les labels en indices
label_names = dataset_df['seafloor']

label_names_unique = label_names.unique()
le = preprocessing.LabelEncoder()
le.fit(label_names_unique)
label_indices = le.transform(label_names)


# rajout d'une dimension chanel pour coller au formalisme de tensorflow
images = images[...,np.newaxis]

#recuperation des metadonnées sur les images
img_shape = images.shape[1:3]
img_count = images.shape[0]
num_channels =  images.shape[3]
num_classes= np.amax(label_indices) + 1

print('Images shape: {}'.format(img_shape))
print('Descriptor Count: {} per images'.format(img_shape[0]*img_shape[1]))
print('Color channels: {}'.format(num_channels))
print('Classes Count: {} ({})'.format(num_classes, label_names_unique))

plot_batch(dataset_df, 3,3,200,200)





labels_one_hot = np.zeros((img_count, num_classes))
imag_line = np.zeros((img_count, img_shape[0] * img_shape[1]))
for i in range(img_count):
    labels_one_hot[i, label_indices[i]] = 1 #Encodage des labels en One Hot
    imag_line[i, :] = images[i].reshape((img_shape[0] * img_shape[1])) #conversion des image de 200*200 en en ligne de 40000 * 1


if np.amax(imag_line) > 1.0: #si les image sont au format 0-255
    imag_line = imag_line / 255 #transformation au format 0-1
    
ds_test = np.zeros((int(img_count/2), img_shape[0] * img_shape[1]))
labels_test = np.zeros((int(img_count/2), num_classes))
ds_train = np.zeros((int(img_count/2), img_shape[0] * img_shape[1]))
labels_train = np.zeros((int(img_count/2), num_classes))
for i in range(img_count): # recuperation de 1 image sur 2 dans chacun des sous groupes
    if i % 2 == 0:
        ds_test[int(i/2), :] = imag_line[i]
        labels_test[int(i/2), :] = labels_one_hot[i]
    else:
        ds_train[int(i/2), :] = imag_line[i]
        labels_test[int(i/2), :] = labels_one_hot[i]
        
print("nombre de d'image par classes({}):".format(label_names_unique))
print(*range(labels_one_hot.shape[1]), sep='\t|')
print(*(np.sum(labels_one_hot, axis=0)), sep='\t|')






























