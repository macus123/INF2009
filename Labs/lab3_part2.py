import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import librosa
import librosa.display
import IPython.display as ipd

df = pd.read_csv("UrbanSound8K.csv")

class_name = ["air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling", "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"]

features = []
labels = []

def parser():
    # Function to load files and extract features
    for i in range(df.shape[0]):
        file_name = '../input/urbansound8k/fold' + str(df["fold"][i]) + '/' + df["slice_file_name"][i]
        # Here kaiser_fast is a technique used for faster extraction
        data, sr = librosa.load(file_name, res_type='kaiser_fast') 
        # We extract mfcc feature from data
        mels = np.mean(librosa.feature.melspectrogram(y = data, sr = sr), axis=1)        
        features.append(mels)
        labels.append(df["classID"][i])

    return features, labels

x, y = parser()
X = np.array(x)
Y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 1)

forest = RandomForestClassifier()

# fit classifier to training set
forest.fit(X_train, y_train)

forest_pred = forest.predict(X_test)

print(classification_report(y_test, forest_pred))

confusion_matrix(Y_test, forest_pred)

plt.figure(figsize = (12, 10))
sns.heatmap(confusion_matrix(Y_test, forest_pred), 
            annot = True, linewidths = 2, fmt="d", 
            xticklabels = class_name,
            yticklabels = class_name)
plt.show()