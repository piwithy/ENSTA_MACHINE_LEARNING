import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from pythonTools import plot_batch, load_batch
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
images2d = np.array([plt.imread(img) for img in dataset_df['image_path'].values.tolist()])
label_names = dataset_df['seafloor'].to_numpy()

# transformer les labels en indices
label_names = dataset_df['seafloor']

label_names_unique = label_names.unique()
le = preprocessing.LabelEncoder()
le.fit(label_names_unique)
label_indices = le.transform(label_names)

# recuperation des metadonnées sur les images
img_shape = images2d.shape[1:3]
img_count = images2d.shape[0]
num_channels = images2d.shape[3] if len(images2d.shape) > 3 else 1
num_classes = np.amax(label_indices) + 1

# convertion des images (200*200) en niveau de gris en un vecteur 40000 * 1
if num_channels == 1:
    images1d = np.array([img.reshape((img_shape[0] * img_shape[1],)) for img in images2d])

print('Images shape: {}'.format(img_shape))
print('Descriptor Count: {} per images'.format(img_shape[0] * img_shape[1]))
print('Color channels: {}'.format(num_channels))
print('Classes Count: {} ({})'.format(num_classes, label_names_unique))

plot_batch(load_batch(dataset_df), 3, 3, 200, 200)  # Plotting d'images aléatoires
