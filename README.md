# Previous Work Samples

This folder contains samples of some work I did for previous companies. 
## 1. AUC 
  This is a Python code that calculates ROC AUC for a working FICO models to make sure that it is working properly for different time segments: yearly, monthly and quarterly. This work was done for USAA in 2017. Since only outcomes in the database were given, and not the model itself, all evalutation metrics had to be calculated manually instead of using sklearn Python package. Pandas and SQL were used to manipulate the data. 
  ### There are 5 models total for 5 different types of transactions:  
  * Credit combined 
  * Credit real time 
  * Credit quasi
  * Credit batch 
  * Debit sign 
  ### Data used: 
  * Cumulative sum of all transactions 
  * Cumulative sum of all valid transactions 
  * Cumulative sum of all fraud transatctions 
  * Cumullative sum of all approved valid transactions 
  * Cumulative sum of all declined transactions due to fraud which were valid 
  * Cumulative sum of all declined non fraud transations which were valid 
  * Cumulative sum of all approved transactions which were fraud 
  * Cumulative sum of all declined fraud transactions which were fraud 
  * Cumulative sum of all declined nonfraud transactions which were fraud 
  * Fraud score type
  * Year, month 
  ### Packages used: 
    collections, datetime, matplotlib, numpy, pandas, pandasql, sys and sklearn.metrics 
  ### Functions 
* **f_bins:**  assignes labels (1-20) based on the Model Score (0-1000)
* **f_nonzero** finds all nonzero values for Model Score
* **data_prep** takes in transactions from the database,calculates cumulative sums for each section calcultate TP, TN, FP and FN and outputs all data needed into a csv file. 
* **calc_ks** calculates Kolmogorov-Smirnof test and plots it 
* **calc_roc** calculates ROC curve and plots it. Returns AUC and gini coefficient 
* **data_prep_addendum** this addendum was done later to incorporate bins and labels into the data preparation
## 2. Random Forest Medical 
  This is a Python code with a model which predicts Juanuvia medication adherenece in diabetic patience given their comborbidities for Merck internal competition in 2018
## 3. RF logreg Marketing
  This is a Python code with a model that predicts sing up for banking products given unbalanced data in 2017
## 4. Reservoir Engineering Proposal
  This is a technical document I wrote where I proposed how to integrate Data Science Operations for Reservor Egineering back in 2015 for Devon Energy
## 5. Projects Overview 
  This is a short pdf presentations of my previous work in oil and gas domain 
## 6. Spotfire Screenshot Examples
  This is a word file with screenshots of one of the dashboards that I built for Parsley Energy in 2016
