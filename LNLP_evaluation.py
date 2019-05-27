# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  : CD-LNLP LOOCV evaluation

import numpy as np
import pandas as pd
import input_and_output
import LNLP_method
import semisupervise


def get_leave_one_out_result(association_data, alpha, neighbor_rate, weight, roc_x_path, roc_y_path, pr_x_path,
                             pr_y_path):
    association_matrix = np.matrix(association_data)
    (rows, cols) = np.where(association_matrix == 1)
    positive_num = len(rows)
    sample_num = positive_num + 1
    k_folds = sample_num
    predict_matrix = np.matrix(np.zeros(association_matrix.shape))

    for k in range(k_folds):
        print("-----------this is %dth cross validation-------------" % (k + 1))
        train_matrix = np.matrix(association_matrix, copy=True)
        if k != (k_folds - 1):
            train_matrix[rows[k], cols[k]] = 0
            score_matrix = LNLP_method.linear_neighbor_predict(train_matrix, alpha,
                                                               neighbor_rate, weight)
            predict_matrix[rows[k], cols[k]] = score_matrix[rows[k], cols[k]]
        else:
            score_matrix = LNLP_method.linear_neighbor_predict(train_matrix, alpha,
                                                               neighbor_rate, weight)
            predict_matrix[np.where(association_matrix == 0)] = score_matrix[np.where(association_matrix == 0)]

    metrics = semisupervise.loo_model_evaluate(association_matrix, predict_matrix, roc_x_path, roc_y_path, pr_x_path,
                                               pr_y_path)
    print(metrics)
    return metrics


if __name__ == '__main__':
    association_data = pd.read_csv('data/association.csv', header=None).values
    result_files = {}

    alphas = np.linspace(0.1, 1.0, 10, dtype="float32")
    neighbor_rates = np.linspace(0.1, 1.0, 10, dtype="float32")
    weights = np.linspace(0, 1.0, 11, dtype='float32')

    for each_alpha in alphas:
        for each_neighbor_rate in neighbor_rates:
            for each_weight in weights:
                temp = str(each_alpha) + '_' + str(each_neighbor_rate) + '_' + str(each_weight)

                temp_loo = temp + '_loo'
                temp_loo_roc_x_key = temp_loo + '_roc_x'
                temp_loo_roc_y_key = temp_loo + '_roc_y'
                temp_loo_pr_x_key = temp_loo + '_pr_x'
                temp_loo_pr_y_key = temp_loo + '_pr_y'

                temp_cv_key = temp + '_5_cv'
                temp_cv_roc_x_key = temp_cv_key + 'roc_x'
                temp_cv_roc_y_key = temp_cv_key + 'roc_y'
                temp_cv_pr_x_key = temp_cv_key + 'pr_x'
                temp_cv_pr_y_key = temp_cv_key + 'pr_y'

                loocv_result_dir = 'evaluation_result/loocv/'
                result_files[temp_loo] = loocv_result_dir + temp_loo + '.csv'
                result_files[temp_loo_roc_x_key] = loocv_result_dir + temp_loo_roc_x_key + '.csv'
                result_files[temp_loo_roc_y_key] = loocv_result_dir + temp_loo_roc_y_key + '.csv'
                result_files[temp_loo_pr_x_key] = loocv_result_dir + temp_loo_pr_x_key + '.csv'
                result_files[temp_loo_pr_y_key] = loocv_result_dir + temp_loo_pr_y_key + '.csv'

                cv_result_dir = 'evaluation_result/5_cv'
                result_files[temp_cv_key] = cv_result_dir + temp_cv_key + '.csv'
                result_files[temp_cv_roc_x_key] = cv_result_dir + temp_cv_roc_x_key + '.csv'
                result_files[temp_cv_roc_y_key] = cv_result_dir + temp_cv_roc_y_key + '.csv'
                result_files[temp_cv_pr_x_key] = cv_result_dir + temp_cv_pr_x_key + '.csv'
                result_files[temp_cv_pr_y_key] = cv_result_dir + temp_cv_pr_y_key + '.csv'

    # obtain evaluation result on some parameter group
    alpha = 0.1
    neighbor_rate = 0.9
    weight = 1.0
    para = str(alpha) + '_' + str(neighbor_rate) + '_' + str(weight)
    loo_metrics_key = para + '_loo'
    loo_roc_x = loo_metrics_key + '_roc_x'
    loo_roc_y = loo_metrics_key + '_roc_y'
    loo_pr_x = loo_metrics_key + '_pr_x'
    loo_pr_y = loo_metrics_key + '_pr_y'
    metrics = get_leave_one_out_result(association_data,
                                       alpha, neighbor_rate, weight, result_files[loo_roc_x], result_files[loo_roc_y],
                                       result_files[loo_pr_x], result_files[loo_pr_y])

    input_and_output.output_evaluation_result(metrics, result_files[loo_metrics_key])
