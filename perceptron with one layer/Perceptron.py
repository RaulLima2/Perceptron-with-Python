"""
MIT License
Copyright (c) 2020 Raul Bruno Santos
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from linear_regression import linear_regression
import csv


class Perceptron(object):
    def __init__(self, no_of_inputs, threshold = 100, learning_rate = 0.9):
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.weights = np.random.rand(no_of_inputs + 1)
        self.no_of_inputs = no_of_inputs
        self.error = 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if np.any(summation >= 0):
            activiation = 1
        else:
            activiation = -1
        return activiation
    
    def train(self, training_inputs, labels):

        for _ in range(self.threshold):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
                
                if prediction != label:
                    self.error += 1/(self.threshold * self.no_of_inputs)

                print('Epoch =%d, Learning Rate = %.3f, Error = %.3f' % (_, self.learning_rate, self.error))
    
    def data_plot(self, inputs, training_inputs, data_size):
        
        linear = linear_regression(inputs, training_inputs, data_size)

        slope = linear.slope
        intercept = linear.intercept

        plt.scatter(inputs, training_inputs, marker='o', label='dates')
        plt.plot(inputs, slope*inputs + intercept, 'r--', label='decision boundary')
        
        ##plt.plot(inputs, training_inputs, marker='x')
        plt.title('Classificação')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

    def write_result(self):

        name_data = 'data_result.csv'
        

        field_name = ['Learning_Rate', 'Verified_data', 'Adjustment']
        dict_date = [{'Learning_Rate' : self.learning_rate, 'Verified_data' : self.no_of_inputs, 'Adjustment' : self.error}]

        with open(name_data, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = field_name)
            writer.writeheader()
            writer.writerows(dict_date)

        name_data = 'data_weights.csv'
        list_date = list(zip(self.weights[1:50],self.weights[51:100],self.weights[101:150], self.weights[151:202]))
        np.savetxt(name_data, list_date, delimiter=',', fmt='%f')