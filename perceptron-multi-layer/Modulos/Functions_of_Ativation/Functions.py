import math 
import numpy as np

def sigmoid(x):
  f = 1 / (1 + math.e**(-x))
  return f

def reLU(x):
	return max(0.0,x)

def tanH(x):
	return 2*sigmoid(2*x) - 1

def leaky_reLU(x):
  alfa = 0.000001

  return max(x*alfa, x)

def d_func_sigmoid(x):

	derivade_f =  sigmoid(x)*(1 - sigmoid(x))

	return derivade_f

def d_func_reLU(x):

  derivade_f = 0.01

  if x > 0:
	  derivade_f =  1
  elif x < 0:
    derivade_f = 0
    
  return derivade_f

def d_func_tanH(x):

	derivade_f =  1 + math.pow(tanH(x),2)

	return derivade_f

def d_func_leaky_reLU(x):
  alpha = 0.00001

  if x > 0:
    return 1
  elif x < 0:
    return alpha