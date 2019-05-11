from itertools import combinations_with_replacement as cwr
from Classifier import *
import numpy as np
import concurrent.futures
from time import time



class Generation:
    def __init__(self, data,number_of_individuals, polynomial_degree, 
            mutation_rate, classifiers = []):
        self.number_of_individuals = number_of_individuals
        self.polynomial_degree = polynomial_degree
        self.data = data
        self.mutation_rate = mutation_rate
        self.percentage_intervals = self.calculate_percentage_interval()
        if classifiers == []:
            self.classifiers = self.create_first_generation()
        else:
            self.classifiers = classifiers
            self.give_mutation()
        self.average_accuracy = self.calculate_average_accuracy()

    def create_first_generation(self):
        classifiers = []
        factors_length = Classifier.calculate_polynomial_length(
            self.data.number_of_features,
            self.polynomial_degree
        )
        if self.data.number_of_classes == 2:
            factors_shape = (self.number_of_individuals,
                factors_length, 
                1)
            factors = np.random.random_sample(factors_shape)
            for i in range(self.number_of_individuals):
                classifiers += [BinaryClassifier(
                    self.data, self.polynomial_degree, factors[i]
                )]
                
        else:
            factors_shape = (self.number_of_individuals,
                factors_length,
                self.data.number_of_classes)
            factors = np.random.random_sample(factors_shape)
            for i in range(self.number_of_individuals):
                classifiers += [MultivariateClassifier(
                    self.data, self.polynomial_degree, factors[i]
                )]
        return sorted(classifiers, 
                    key = lambda cl: cl.accuracy, reverse = True)

    def calculate_percentage_interval(self):
        percentage_intervals = [self.number_of_individuals]
        for i in range(self.number_of_individuals - 1, 0, -1):
           percentage_intervals.append(percentage_intervals[-1] + i)
        return percentage_intervals

    def select_index_from_individuals(self):
        total_score = self.number_of_individuals * (self.number_of_individuals + 1) / 2
        rand_int = np.random.random_integers(1, total_score)
        for i in range(len(self.percentage_intervals)):
            if rand_int <= self.percentage_intervals[i]:
                return i
            else:
                rand_int = np.random.random_integers(1, total_score)     

    def select_two_parents(self):
        first_index = self.select_index_from_individuals()
        second_index = self.select_index_from_individuals()
        while first_index == second_index:
            second_index = self.select_index_from_individuals()
        return self.classifiers[first_index], self.classifiers[second_index] 

    @staticmethod
    def cross_over(first_parent, second_parent):
        cross_over_lines = np.random.random_integers(
            low=0, high=first_parent.factors.shape[0] - 1, size=first_parent.factors.shape[1]
        )
        first_child_factors = np.empty(shape=first_parent.factors.shape)
        second_child_factors = np.empty(shape=second_parent.factors.shape)
        for i in range(len(cross_over_lines)):
            first_child_factors[:,i] =  np.append(
                first_parent.factors[:cross_over_lines[i], i], 
                second_parent.factors[cross_over_lines[i]:, i])
            second_child_factors[:,i] = np.append(
                second_parent.factors[:cross_over_lines[i], i], 
                first_parent.factors[cross_over_lines[i]:, i])
            
        first_child_factors = np.array(first_child_factors)
        second_child_factors = np.array(second_child_factors)
        if first_parent.data.number_of_classes == 2:
            first_child = BinaryClassifier(
                first_parent.data, first_parent.polynomial_degree, first_child_factors
            )
            second_child = BinaryClassifier(
                first_parent.data, first_parent.polynomial_degree, second_child_factors
            )
        else:
            first_child = MultivariateClassifier(
                first_parent.data, first_parent.polynomial_degree, first_child_factors
            )
            second_child = MultivariateClassifier(
                first_parent.data, first_parent.polynomial_degree, second_child_factors
            )
        return first_child, second_child

    @staticmethod
    def init_from_old_generation(old):
        new_classifiers = []
        for i in range(0, old.number_of_individuals, 2):
            first_parent, second_parent = old.select_two_parents()
            first_child, second_child = Generation.cross_over(first_parent, second_parent)
            new_classifiers += [first_child, second_child]
        new_classifiers = np.array(sorted(new_classifiers, 
            key=lambda c:c.accuracy, reverse=True
        ))
        return Generation(old.data, old.number_of_individuals, old.polynomial_degree,
                    old.mutation_rate, new_classifiers)

    def calculate_average_accuracy(self):
        total = 0
        for cls in self.classifiers:
            total += cls.accuracy
        return total / self.number_of_individuals

    def give_mutation(self):
        for classifier in self.classifiers:
            random = np.random.random_sample(classifier.factors.shape)
            for i in range(random.shape[0]):
                for j in range(random.shape[1]):
                    if random[i, j] <= self.mutation_rate:
                        classifier.factors[i, j] = np.random.rand()

