from ML import mlp as mlp
import numpy as np
from Modulos.Functions_of_Ativation import Functions as f

array_n = [0.001, 10, 2, 3, 1]
p = mlp.perceptron_mlp(array_n, f.leaky_reLU, f.d_func_leaky_reLU)

#Xor Problem
input_xor = np.matrix([[1,1],[1,0],[0,1],[0,0]])
output_xor = np.matrix([[0],[1],[1],[0]])
labels_xor = np.matrix([[1,1],[1,0],[0,1],[0,0]])

p.train(labels_xor, output_xor)

print(p.prediction(input_xor[0,:]), (labels_xor[0,:]))
print(p.prediction(input_xor[1,:]), (labels_xor[1,:]))
print(p.prediction(input_xor[2,:]), (labels_xor[2,:]))
print(p.prediction(input_xor[3,:]), (labels_xor[3,:]))

print(p.weights_hidden_i)
print(p.weights_hidden_o)