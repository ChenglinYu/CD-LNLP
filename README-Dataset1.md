# Case Study on Luo's dataSet(our golden dataset)

## Dataset
- `Dataset1/association.csv` is the circRNA-disease association matrix,our golden dataset, which contains 331 associations between 312 circRNAs and 40 diseases.
- `Dataset1/all_circRNAs.csv` contains all the circRNAs, corresponding to the rows of the association matrix.
- `Dataset1/all_diseases.csv` contains all the diseases, corresponding to the columns of the association matrix.

## Run case_study.py

When you run `case_study.py`, you will get the score matrix in the `case_study_scores` folder. We have produced the score matrix, that is `case_study_scores/Dataset1_scores.csv`.

For every disease, the candidate circRNAs are in the text file named as the disease's name in `Dataset1_result/disease` folder in descending order of score.