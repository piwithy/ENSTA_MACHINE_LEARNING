import usefulCmds
from pythonTools import plot_batch
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
from sklearn.utils import shuffle


class Preprocessing():
    def __init__(self):
        self.DATASET_PATH = usefulCmds.DATASET_PATH
        self.LABEL_PATH = usefulCmds.LABEL_PATH
        self.ABSOLUT_PATH = usefulCmds.ABSOLUT_PATH
        self.resize = False
        self.dimension = 200
        self.image_grid = {
            "x": 5,
            "y": 5,
        }
        self.image_scale = {
            "x": 120,
            "y": 120,
        }

        self.labellingImage = usefulCmds.label_names.unique()
        self.labellingNumber = usefulCmds.label_indices

        self.X_global = []
        self.y_global = []

        self.X_train = []
        self.y_train = []

        self.X_test = []
        self.y_test = []

        self.X_eval = []
        self.y_eval = []

        self.descripteur = {'Posidonia': 0, 'Ripple 45°': 1, 'Rock': 2, 'Sand': 3, 'Silt': 4, 'Ripple vertical': 5}

        self.image_in_the_matrix = []
        self.verifynormalization = True

        self.max = 0

    def setNormalization(self, value):
        self.verifynormalization = value

    def importingCSV(self):
        self.dataset_df = pd.read_csv(self.LABEL_PATH)
        self.max = len(self.dataset_df)

        self.dataset_df["image_path"] = self.dataset_df.apply(lambda row: (self.DATASET_PATH + row["id"]), axis=1)
        self.dataset_df["image_matrix"] = self.dataset_df.apply(
            lambda row: self.normalizationImage(cv2.imread(self.DATASET_PATH + row["id"]), False), axis=1)

        self.dataset_df["descripteur"] = [self.descripteur[item] for item in self.dataset_df["seafloor"]]

    def getSizeImage(self):
        return self.dimension

    def loadingImage(self):
        return usefulCmds.feature_values

    def plotingImage(self):
        pythonTools.plot_batch(self.dataset_df, self.image_grid["x"], self.image_grid["y"], self.image_scale["x"],
                               self.image_scale["y"])

    def getLabels(self):
        return self.labellingImage

    def getNumberofLabels(self):
        return len(self.descripteur) + 1

    def getXandYGlobal(self):
        return self.X_global, self.y_global

    def setResizeImage(self, resize=True, dimension=15):
        self.resize = resize
        self.dimension = dimension

    def getLayerSize(self):
        if self.resize:
            return self.dimension * self.dimension
        else:
            return 400 * 400

    def normalizationImage(self, image, color=True):
        if color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.resize:
            image = cv2.resize(image, (self.dimension, self.dimension))

        if self.verifynormalization:
            image = cv2.equalizeHist(image)
            image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = image.flatten()
        return image

    def generatingXandY(self):
        dataset_suffle = shuffle(self.dataset_df)
        dataset_suffle = dataset_suffle.reset_index()

        split_1 = round(self.max / 3 * 2)
        split_2 = split_1 + round(self.max / 6)

        for i in range(self.max):
            self.X_global.append(dataset_suffle["image_matrix"][i])
            self.y_global.append([dataset_suffle["descripteur"][i]])
            if i < split_1:
                self.X_train.append(dataset_suffle["image_matrix"][i])
                self.y_train.append([dataset_suffle["descripteur"][i]])
            elif i > split_2:
                self.X_test.append(dataset_suffle["image_matrix"][i])
                self.y_test.append([dataset_suffle["descripteur"][i]])
            else:
                self.X_eval.append(dataset_suffle["image_matrix"][i])
                self.y_eval.append([dataset_suffle["descripteur"][i]])
        pass

    def getXandY(self):
        return np.array(self.X_train), np.array(self.X_eval), np.array(self.X_test), np.array(self.y_train), np.array(
            self.y_eval), np.array(self.y_test)

    def matFileImporter(self, name):
        data = loadmat(self.ABSOLUT_PATH + "/dataset/" + name)
        _, _, _, X = data.keys()
        self.max = len(X)

        for i in range(self.max):
            self.dataset_df["image_matrix"][i] = self.normalizationImage(np.c_[np.ones((data[X].shape[0], 1)), data[X]],
                                                                         color=False)
