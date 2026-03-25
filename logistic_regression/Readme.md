# README

## Overview

This notebook analyzes a drug consumption dataset and builds multiple machine learning models to classify drug usage. It includes:

* Exploratory data analysis (EDA)
* Logistic regression (implemented from scratch)
* Model evaluation using classification metrics
* ROC curve and AUC computation
* Feature selection (forward selection)
* Regularization and feature scaling
* Neural network modeling using `sklearn`

---

## Dataset

The notebook expects a dataset file named:

```
drug_consumption.data
```

### Required Format

* Comma-separated values
* Columns must match the following order:

```
ID, Age, Gender, Education, Country, Ethnicity,
Nscore, Escore, Oscore, Ascore, Cscore,
Impulsive, SS,
Alcohol, Amphet, Amyl, Benzos, Caff, Cannabis,
Choc, Coke, Crack, Ecstasy, Heroin, Ketamine,
Legalh, LSD, Meth, Mushrooms, Nicotine, Semer, VSA
```
---

## Requirements

Install the following Python libraries before running:

```
pandas
numpy
matplotlib
scikit-learn
```

### Installation (pip)

```
pip install pandas numpy matplotlib scikit-learn
```

---

## Execution Steps

### Jupyter Notebook

1. Open VS Code
2. Open the notebook file.
3. Run all cells sequentially using preferred kernel.

---

## Author

Diana Johnson

