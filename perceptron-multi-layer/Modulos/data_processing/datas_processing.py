import pandas as pd
import numpy as np

def data_processings(name_file):
	Dataframe = pd.read_csv(name_file)

	input = np.matrix(Dataframe['x'].to_numpy())
	output = np.matrix(Dataframe['y'].to_numpy())
	labels = np.matrix(Dataframe['classificacao'].to_numpy())

	for i in range(input.size):
		if labels[0,i] == -1.0:
			labels[0,i] = 0.0
	
	return input, output, labels