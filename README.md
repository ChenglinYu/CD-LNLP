# Case Study on Luo's dataSet(our golden dataset)

## Dataset
- `data/association.csv` is the circRNA-disease association matrix,our golden dataset, which contains 331 associations between 312 circRNAs and 40 diseases.
- `data/all_circRNAs.csv` contains all the circRNAs, corresponding to the rows of the association matrix.
- `data/all_diseases.csv` contains all the diseases, corresponding to the columns of the association matrix.

## Run case_study.py

When you run `case_study.py`, you will get the score matrix in the `produced_data` folder. We have produced the score matrix, that is `produced_data/scores.csv`.

For every disease, the candidate circRNAs are in the text file named as the disease's name in `result/disease` folder in descending order of score.