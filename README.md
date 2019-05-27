# Case Study on Lei's dataSet(the up-to-date dataset)

## Dataset
- `data/association.csv` is the up-to-date circRNA-disease association matrix, which contains 650 associations between 603 circRNAs and 88 diseases.
- `data/all_circRNAs.csv` contains all the circRNAs, corresponding to the rows of the association matrix.

- `data/all_diseases.csv` contains all the diseases, corresponding to the columns of the association matrix.


## Run case_study.py

When you run `case_study.py`, you will get the score matrix in the `produced_data` folder. We have produced the score matrix, that is `produced_data/scores.csv`.

For every disease, the candidate circRNAs are in the text file named for the disease's name in `result/disease` folder.