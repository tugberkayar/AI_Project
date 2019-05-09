from itertools import combinations_with_replacement as cwr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from abc import ABC, abstractmethod


class Classifier:
    def __init__(self, data, 
            polynomial_degree, factors):
        self.data = data
        self.polynomial_degree = polynomial_degree
        self.factors = factors
    
    @staticmethod
    def calculate_polynomial_result(factors, variables, degree):
        number_of_features = len(variables)
        combinations = []
        for d in range(degree, 0, -1):
            combinations += list(cwr(range(number_of_features), d))
        i = 0
        sum = 0
        for element in combinations:
            mul = 1
            for j in element:
                mul *= variables[j]
            mul *= factors[i]
            sum += mul
            i += 1
        return sum + factors[-1] #including the constant

    @staticmethod
    def calculate_polynomial_length(features, degree):
        length = 0
        for d in range(1, degree + 1):
            temp = len(list(cwr(range(features), d)))
            length += temp
        return length + 1 #including the constant
    
    @staticmethod
    def init_factors(shape):
        return np.random.random_sample(shape)
    
    @abstractmethod
    def estimate(self):
        pass

    @abstractmethod
    def calculate_accuracy(self):
        pass


class BinaryClassifier(Classifier):
    def __init__(self, data, 
            polynomial_degree, factors = []):
        if factors == []:
            shape_of_factors = (Classifier.calculate_polynomial_length(
                data.number_of_features, polynomial_degree), 1)
            Classifier.__init__(self, data, polynomial_degree, 
                Classifier.init_factors(shape_of_factors))
        else:
            Classifier.__init__(self, data, polynomial_degree, factors)
        self.estimations = self.estimate()
        self.accuracy = self.calculate_accuracy()
    

    def estimate(self):
        estimations = np.empty(shape = (self.data.number_of_samples, ))
        for row, index in self.data.features.iterrows():
            estimations[row] = Classifier.calculate_polynomial_result(
                    self.factors, index.values, self.polynomial_degree
            )
        sc = MinMaxScaler()
        estimations = sc.fit_transform(estimations.reshape(-1,1))
        return estimations

    def calculate_accuracy(self):
        i = 0
        correct = 0
        for result in self.data.target:
            if result == 1 and self.estimations[i] >= 0.5:
                correct += 1
            elif result == 0 and self.estimations[i] < 0.5:
                correct += 1
            i += 1
        return correct / self.data.number_of_samples
