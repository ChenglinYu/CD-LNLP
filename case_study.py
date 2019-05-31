# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  : output case study score matrix

import LNLP_method
import numpy as np
import pandas as pd


# write matrix to csv file
def matrix_to_csv(matrix, file_name):
    arrays = np.array(matrix)
    arrays_frame = pd.DataFrame(arrays)
    arrays_frame.to_csv(file_name, header=None, index=None)


if __name__ == '__main__':
    association_data1 = pd.read_csv('Dataset1/association.csv', header=None).values
    score_matrix1 = LNLP_method.linear_neighbor_predict(association_data1, alpha=0.1, neighbor_rate=0.9,
                                                        circRNA_weight=1.0)
    score_matrix1[np.where(association_data1 == 1)] = 0
    matrix_to_csv(score_matrix1, 'case_study_scores/Dataset1_scores.csv')
    association_data2 = pd.read_csv('Dataset2/association.csv', header=None).values
    score_matrix2 = LNLP_method.linear_neighbor_predict(association_data2, alpha=0.1, neighbor_rate=0.9,
                                                        circRNA_weight=1.0)
    score_matrix2[np.where(association_data2 == 1)] = 0
    matrix_to_csv(score_matrix2, 'case_study_scores/Dataset2_scores.csv')
