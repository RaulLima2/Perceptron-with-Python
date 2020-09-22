import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron

training = pd.read_csv('dados/dados_2D_parabola.csv')

training_x = training['x'].to_numpy()
training_y = training['y'].to_numpy()
labels = training['classificacao'].to_numpy()


x = np.array(training_x)
y = np.array(training_y) 

perceptron = Perceptron(201)


perceptron.train(y, labels)

##perceptron.train(training_inputs, labels)

perceptron.predict(x)


perceptron.data_plot(x, y, 201)

perceptron.write_result()