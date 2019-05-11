from Data import Data
from Classifier import BinaryClassifier
import pandas as pd
from Classifier import Classifier
import numpy as np

df = pd.read_csv("heart.csv")
data = Data(data=df, columns=df.columns,
            target_col='target',
            columns_to_scale=df.columns[:-1])




from Generation import Generation
gnr = Generation(data, 100, 1, 0.2)

print(gnr.classifiers[0].factors.shape)
i = 0
while i<100:
    print(i, ". GENERATION")
    print("AVERAGE:", gnr.average_accuracy)
    print("BEST:",gnr.classifiers[0].accuracy)
    gnr = Generation.init_from_old_generation(gnr)
    i += 1
    print("\n\n\n")