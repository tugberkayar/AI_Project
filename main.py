from Data import Data
from Classifier import BinaryClassifier
import pandas as pd
from Classifier import Classifier
import numpy as np

df = pd.read_csv("heart.csv")
data = Data(data=df.values, columns=df.columns,
            target_col='target',
            columns_to_scale=df.columns[:-1])



factors = np.random.rand(data.number_of_features)
cls = BinaryClassifier(factors, data, 1)
print(cls.accuracy)