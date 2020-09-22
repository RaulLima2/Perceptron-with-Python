import numpy as np

class linear_regression(object):
    def __init__(self, input_x, output_y, data_size):
        self.slope = 0
        self.intercept = 0
        
        m_x = np.mean(input_x)
        m_y = np.mean(output_y)

        SS_xy = np.sum(input_x * output_y) - data_size * m_y*m_x
        SS_xx = np.sum(input_x*input_x) - data_size * m_x * m_x

        

        self.slope = SS_xy / SS_xx

        self.intercept = m_y - self.slope*m_x