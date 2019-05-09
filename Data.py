import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Data:
    def __init__(self, columns, data, 
            target_col, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        data = pd.DataFrame(columns = columns, data = data)
        self.features = data.drop([target_col], axis = 'columns')
        self.target = data[target_col]
        self.number_of_features = len(self.features.columns)
        self.number_of_samples = len(data.index)
        self.number_of_classes = len(data.groupby(data[target_col]))
        self.sc = self.scale_data()

    def scale_data(self):
        sc = MinMaxScaler()
        self.features[self.columns_to_scale] = sc.fit_transform(
            self.features[self.columns_to_scale])
        return sc

    def inverse_scale(self):
        self.features[columns_to_scale] = self.sc.inverse_transform(
            self.features[self.columns_to_scale])