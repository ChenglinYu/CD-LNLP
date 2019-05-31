# -*- coding: utf-8 -*-
# @Author: chenglinyu
# @Date  : 2019/3/2
# @Desc  : CD-LNLP LOOCV evaluation

import LNLP_method
import numpy as np
import pandas as pd
import csv


# write matrix to csv file
def matrix_to_csv(matrix, file_name):
    arrays = np.array(matrix)
    arrays_frame = pd.DataFrame(arrays)
    arrays_frame.to_csv(file_name, header=None, index=None)


# write evaluation result to file
def output_evaluation_result(evaluation_result, file_name):
    with open(file_name, "w") as csvfile:
        writer = csv.writer(csvfile)
        # write columns_name
        writer.writerow(["aupr", "auc", "f1_score", "accuracy", "recall", "specificity", "precision"])
        writer.writerow(evaluation_result)


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

    metrics = loo_model_evaluate(association_matrix, predict_matrix, roc_x_path, roc_y_path, pr_x_path,
                                 pr_y_path)
    print(metrics)
    return metrics


def loo_model_evaluate(interaction_matrix, predict_matrix, roc_x_path, roc_y_path, pr_x_path, pr_y_path):
    real_score = np.matrix(np.array(interaction_matrix).flatten())
    predict_score = np.matrix(np.array(predict_matrix).flatten())
    metrics = get_metrics(real_score, predict_score, roc_x_path, roc_y_path, pr_x_path, pr_y_path)
    return metrics


def get_metrics(real_score, predict_score, roc_x_path, roc_y_path, pr_x_path, pr_y_path):
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))  # 进行了去除重复
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.array(range(1, 1000)) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1

    TP = predict_score_matrix * real_score.T
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    matrix_to_csv(x_ROC, roc_x_path)
    matrix_to_csv(y_ROC, roc_y_path)
    auc = 0.5 * (x_ROC[1:] - x_ROC[:-1]).T * (y_ROC[:-1] + y_ROC[1:])

    recall_list = tpr
    precision_list = TP / (TP + FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack((recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    matrix_to_csv(x_PR, pr_x_path)
    matrix_to_csv(y_PR, pr_y_path)
    aupr = 0.5 * (x_PR[1:] - x_PR[:-1]).T * (y_PR[:-1] + y_PR[1:])

    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index, 0]
    accuracy = accuracy_list[max_index, 0]
    specificity = specificity_list[max_index, 0]
    recall = recall_list[max_index, 0]
    precision = precision_list[max_index, 0]
    return [aupr[0, 0], auc[0, 0], f1_score, accuracy, recall, specificity, precision]


if __name__ == '__main__':
    association_data = pd.read_csv('Dataset1/association.csv', header=None).values
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

    output_evaluation_result(metrics, result_files[loo_metrics_key])
