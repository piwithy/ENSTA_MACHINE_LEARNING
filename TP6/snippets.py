import csv
import os
from PIL import Image
import numpy as np
from pythonTools import plot_batch
import pandas as pd

labels_csv_file = open(os.path.join("dataset", "labels", "labels.csv"), 'r')  # ouverture du fichier de label
lines = csv.reader(labels_csv_file)
labels = []
n = 0
next(lines)
for line in lines:
    data = {"row": n, "id": line[0], "seafloor": line[1]}
    n += 1
    labels.append(data)

X = np.zeros((len(labels), 40000))
for i in range(len(labels)):
    image = Image.open(os.path.join("dataset", "imgs", labels[i]['id']))
    data_array = np.asarray(image).reshape((1, 40000))
    X[i, :] = data_array

print("loaded {} images w/{} descriptor each".format(X.shape[0], X.shape[1]))

plot_batch(pd.DataFrame(labels), 3, 3, 200, 200)
