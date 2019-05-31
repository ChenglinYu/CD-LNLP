Dataset and Code for "Predicting CircRNA-disease Associations through Linear Neighborhood Label Propagation Method".

# Dataset

## Dataset1
- `Dataset1/association.csv` is the circRNA-disease association matrix of `Dataset1`, which contains 331 associations between 312 circRNAs and 40 diseases.
- `Dataset1/all_circRNAs.csv` contains all the circRNAs, corresponding to the rows of the association matrix.
- `Dataset1/all_diseases.csv` contains all the diseases, corresponding to the columns of the association matrix.

## Dataset2
- `Dataset2/association.csv` is the circRNA-disease association matrix of `Dataset2`, which contains 650 associations between 603 circRNAs and 88 diseases.
- `Dataset2/all_circRNAs.csv` contains all the circRNAs, corresponding to the rows of the association matrix.
- `Dataset2/all_diseases.csv` contains all the diseases, corresponding to the columns of the association matrix.

# Code


- `case_study.py`  calculates score matrices of case studies on Dataset1 and Dataset2 respectively.

-  `LNLP_method.py` contains our method function, that is `linear_neighbor_predict`.

- `LNLP_evaluation.py` implements LOOCV of CD-LNLP on `Dataset1`.



# Result

- `case_study_scores`
    - `Dataset1_scores.csv` is the score matrix of case study on `Dataset1`.
    - `Dataset2_scores.csv` is the score matrix of case study on `Dataset2`.

- `Dataset1_result/disease`

    For every disease in `Dataset1`, the candidate circRNAs are in the text file named as the disease's name in `Dataset1_result/disease` folder in descending order of score.

- `Dataset2_result/disease`

    For every disease in Dataset2, the candidate circRNAs are in the text file named as the disease's name in `Dataset2_result/disease` folder in descending order of score.

- `evaluation_result/loocv`

    `evaluation_result/loocv` contains our method's evaluation result on LOOCV.
    
    - `0.1_0.9_1.0_loo.csv` contains the values of 6 metrics.
    - `0.1_0.9_1.0_loo_pr_x.csv` contains the values of **recall** on different thresholds.
    - `0.1_0.9_1.0_loo_pr_y.csv` contains the values of **precision** on different thresholds.
    - `0.1_0.9_1.0_loo_roc_x.csv` contains the values of **False Positive Rate** on different thresholds.
    - `0.1_0.9_1.0_loo_roc_y.csv` contains the values of **True Positive Rate** on different thresholds.


