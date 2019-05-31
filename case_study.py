# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  :

import LNLP_method
import numpy as np
import pandas as pd
import csv

# write matrix to csv file
def matrix_to_csv(matrix, file_name):
    arrays = np.array(matrix)
    arrays_frame = pd.DataFrame(arrays)
    arrays_frame.to_csv(file_name, header=None, index=None)


if __name__ == '__main__':
    association_data = pd.read_csv('Dataset2/association.csv', header=None).values
    score_matrix = LNLP_method.linear_neighbor_predict(association_data,alpha=0.1,neighbor_rate=0.9,
                                                       circRNA_weight=1.0)
    score_matrix[np.where(association_data == 1)] = 0
    matrix_to_csv(score_matrix, 'case_study_scores/Dataset2_scores.csv')
