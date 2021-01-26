import numpy as np
import numpy.matlib


class matrix(object):
    def matrix(rows, columns):
        new_matrix = 2 * np.matlib.rand(rows, columns) - 1
        return new_matrix
    
    def fromArray(array):
      new_matrix = np.matrix(array).T
      return new_matrix

    def sum(matrix_1, matrix_2):
        new_matrix = matrix_1 + matrix_2
        return new_matrix

    def sub(matrix_1, matrix_2):
        new_matrix = matrix_1 - matrix_2
        return new_matrix

    def mult(matrix_1, matrix_2):
        new_matrix = np.dot(matrix_1, matrix_2)
        return new_matrix

    def scalar(matrix_1, k):
        new_matrix = np.multiply(matrix_1, k)
        return new_matrix

    def print_matrix(matrix_1):
        print(matrix_1)

    def transpost(matrix_1):
        return matrix_1.transpose()

    def product_hadamard(matrix_1, matrix_2):
        new_matrix = np.multiply(matrix_1, matrix_2)
        return new_matrix

    def map(func, matrix_1):
        fv = np.vectorize(func)
        new_matrix = fv(matrix_1)
        return new_matrix
