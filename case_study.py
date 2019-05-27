# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  : output case study score matrix

import pandas as pd
import LNLP_method
import numpy as np
import input_and_output

if __name__ == '__main__':
    association_data = pd.read_csv('data/association.csv', header=None).values
    score_matrix = LNLP_method.linear_neighbor_predict(association_data, alpha=0.1, neighbor_rate=0.9,
                                                       circRNA_weight=1.0)
    score_matrix[np.where(association_data == 1)] = 0
    input_and_output.matrix_to_csv(score_matrix, 'produced_data/scores.csv')
