# Case Study on Lei's dataSet(the up-to-date dataset)

## Dataset
- `Dataset2/association.csv` is the up-to-date circRNA-disease association matrix, which contains 650 associations between 603 circRNAs and 88 diseases.
- `Dataset2/all_circRNAs.csv` contains all the circRNAs, corresponding to the rows of the association matrix.

- `Dataset2/all_diseases.csv` contains all the diseases, corresponding to the columns of the association matrix.


## Run case_study.py

When you run `case_study.py`, you will get the score matrix in the `case_study_scores` folder. We have produced the score matrix, that is `case_study_scores/Dataset2_scores.csv`.

For every disease, the candidate circRNAs are in the text file named as the disease's name in `Dataset2_result/disease` folder in descending order of score.