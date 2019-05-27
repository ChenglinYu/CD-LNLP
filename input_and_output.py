# -*- coding: utf8 -*-
import numpy as np
import pandas as pd
import csv


# write matrix to csv file
def matrix_to_csv(matrix, file_name):
    arrays = np.array(matrix)
    arrays_frame = pd.DataFrame(arrays)
    arrays_frame.to_csv(file_name, header=None, index=None)


# write evaluation result to file
def output_evaluation_result(evalutaion_result, file_name):
    with open(file_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # write column name
        writer.writerow(["aupr", "auc", "f1_score", "accuracy", "recall", "specificity", "precision"])
        writer.writerow(evalutaion_result)


if __name__ == '__main__':
    print('Todo')
