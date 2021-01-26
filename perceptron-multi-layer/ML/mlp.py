from Modulos.Matrix import matrix as m
import numpy as np
import math



class perceptron_mlp(object):
    #array_n serve para inserir os parametros necessarios para funcionamento
    # da rede neural perceptron de multiplas camadas
    def __init__(self, array_n, func_ativation, d_func):
        self.error_prediction = [0.0]
        self.predict = []
        self.input = []
        self.hidden = []
        self.output = []

        self.learning_rate = array_n[0]
        self.training_epochs = array_n[1]
        self.n_input = array_n[2]  #quantos valores de entrada?
        self.n_hidden = array_n[3]  #quantos neurônios na camada oculta?
        self.n_output = array_n[4]  #quantos neurônios na camada de sáida?
        self.ativation = func_ativation # função de ativação
        self.d_ativation = d_func #derivade of function ativation

        self.bias_ih = m.matrix.matrix(self.n_hidden, 1)
        self.bias_oh = m.matrix.matrix(self.n_output, 1)

        self.weights_hidden_i = m.matrix.matrix(self.n_hidden, self.n_input)
        self.weights_hidden_o = m.matrix.matrix(self.n_output, self.n_hidden)


    def feedfoward(self, label):
        #Da entrapa para a camada escondida
        input = m.matrix.fromArray(label)
        hidden = m.matrix.mult(self.weights_hidden_i, input)
        hidden = m.matrix.sum(hidden, self.bias_ih)

        hidden = m.matrix.map(self.ativation, hidden)

        # The layerDa camada escondida para a saida
        output = m.matrix.mult(self.weights_hidden_o, hidden)
        output = m.matrix.sum(output, self.bias_oh)
        output = m.matrix.map(self.ativation, output)

        self.output = output
        self.input = input
        self.hidden = np.matrix(hidden)

    def backpropagation(self, array_output):
        # THE OUTPUT -> HIDDEN
        expected = m.matrix.fromArray(array_output)
        output_error = m.matrix.sub(expected, self.output)
        d_output = m.matrix.map(self.d_ativation, self.output)

        hidden_transpost = m.matrix.transpost(self.hidden)

        gradient = m.matrix.product_hadamard(d_output, output_error)
        gradient = m.matrix.scalar(gradient, self.learning_rate)

        ##Ajust bias Output Hidden
        self.bias_oh = m.matrix.sum(self.bias_oh, gradient)

        weights_hidden_o_delta = m.matrix.mult(gradient, hidden_transpost)
        self.weights_hidden_o = m.matrix.sum(self.weights_hidden_o,
                                      weights_hidden_o_delta)

        # THE HIDDEN -> INPUT
        weights_hidden_o_transpost = m.matrix.transpost(self.weights_hidden_o)
        hidden_error = m.matrix.mult(weights_hidden_o_transpost, output_error)
        d_hidden = m.matrix.map(self.d_ativation, self.hidden)
        input_transpost = m.matrix.transpost(self.input)

        gradient_hidden = m.matrix.product_hadamard(hidden_error, d_hidden)
        gradient_hidden = m.matrix.scalar(gradient_hidden, self.learning_rate)

        #Adjust Bias Output Hidden
        self.bias_ih = m.matrix.sum(self.bias_ih, gradient_hidden)

        # Adjust Weigths Hidden Input
        weights_hidden_input_deltas = m.matrix.mult(gradient_hidden, input_transpost)

        self.weights_hidden_i = m.matrix.sum(self.weights_hidden_i,
                                      weights_hidden_input_deltas)

        self.error_prediction = output_error

    def prediction(self, label):
        input = m.matrix.fromArray(label)
        hidden = m.matrix.mult(self.weights_hidden_i, input)
        hidden = m.matrix.sum(hidden, self.bias_ih)

        hidden = m.matrix.map(self.ativation, hidden)

        output = m.matrix.mult(self.weights_hidden_o, hidden)
        output = m.matrix.sum(output, self.bias_oh)
        output = np.array(m.matrix.map(self.ativation, output))

        self.predict.append(output)

        return output

    def train(self, labels, training_inputs):

        if self.n_input == 1:
            self.train_one_dim_input(labels, training_inputs)

        elif self.n_input > 1:
            self.train_mult_layer_input(labels, training_inputs)
       
    
    def train_one_dim_input(self, labels, training_inputs):
        for epoch in range(self.training_epochs):
          for row in range(labels.shape[0]):
              for colunms in range(labels.shape[1]):
                  self.feedfoward(labels[row, colunms])
                  self.backpropagation(training_inputs[row, colunms])
        
        print('Terminou\n')


    def train_mult_layer_input(self, labels, training_inputs):
        for epoch in range(self.training_epochs):
          for row in range(labels.shape[0]):
            self.feedfoward(labels[row, :])
            self.backpropagation(training_inputs[row, :])
      
        print('Terminou\n')

    def mse(self, inputs, predictions):
      result = 00.0
      for i in range(inputs.shape[1]):
        result += (math.pow(abs(inputs[0,i] - predictions[i]),2)/inputs.shape[1])

      return result

      