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
  This is a Python code with a model which predicts risk of nonadherence for Januvia patient cohort resutls. This model was done for Merck internal competition in 2018.
### Introduction
Medication adherence measures the extent to which a patient takes medications as prescribed by their health care provider. Published studies note that up to half of chronically ill patients do not adhere to their medication regiments. The financial impact of non-adherence is estimated at $100 billion annually due to avoidable hospitalizations with other health factors adding to that cost. To increase medication adherence and decrease overall healthcare spend, it is vital that health care providers be able to accurately measure adherence and identify potential gasp in care. 

Data Science competition covered Januvia (Sitagliptin) product for anti-diabetic therapy in DPP4 class to treat type 2 diabetes. 

Data samples included demographic data, Charlson comorbidity index, comorbidity status, health care resource utilization and cost, drug and clinical procedures used in the study period. 
### Definitions
Comorbidity is the presence of one or more additional diseases or disorders co-occurring, with a primary disease or disorder; in the countable sense of the term, a comorbidity is each additional disorder or disease. The additional disorder may be a behavioral or mental disorder. 

The Charlson comorbidity index predicts the one-year mortality for a patient who may have a range of comorbid conditions, such as heart disease, AIDS, or cancer (a total of 22 conditions). Each condition is assigned a score of 1,2,3 or 6, depending on the risk of dying associated with each one. Scores are summed to provide a total score to predict mortality. 
### Methodology
#### Step 1: Hypothesis: 
Due to the time constraint the hypotheses were not formed or tested. Instead the research data on medication adherence was used. The columns were selected based on the research made by medical professionals. The columns were selected based on the criteria below: 
    1. Health literacy
    2. Literacy
    3. Treatment cost 
    4. Age
    5. Ethnicity
    6. Prescription fill rates (Medication Possession Ratio)
    7. Course completion
    8. Health and disease management
In comorbidity data, only records with heart diseases and psychiatric diseases were kept. Research indicated that these diseases have the highest effect on the medication adherence. In addition, all diabetes data along with its complications were added as well. 
Many columns were not present in the Data Dictionary. Hence Spearman and Pearson coefficients were used to exclude all unnecessary columns. One column in each pair that had a coefficient higher than 0.65, was excluded. 
#### Step 2: Cleaning and Transforming the Data
 * Selected columns from data tables were merged using Pandas on patient ID using inner join. Pandas tends to replicate rows during merging. When this happens, it is useful to use drop_duplicates in the merge function. 
 * All null columns were dropped using dropna function in Pandas. If the data had 10% of the null values, the rows were dropped. If the data had more than 10% of the missing data, the columns were dropped. This method was revisited and instead all columns with the missing data were dropped (Andrew’s suggestion). 
 * Three categorical data columns with demographic information were transformed using get_dummies function. At first Label Encoder was considered to transform categorical values instead. Andrew suggested to use dummy variable’s instead. This method to be further investigated. I found a Kaggle paper on Label Encoding vs Dummy Variables. 
 * The data was examined using Seaborn. It was determined that the data was unbalanced. Further research showed that the disease data is highly unbalanced. The solution to this would be oversampling the low variable class and under sampling highly occurring class. Due to time constraints this method was abandoned. 
The procedures above were done to clean and transform both train and test data. 
#### Step 3: Model Selection
Two targets were given: a continuous variable and a class variable for PPD. Regression was used to predict a continuous variable. Since the data was unbalanced, and not treated, regression yielded a high error and was abandoned. Random forest was used instead to predict the classifier target. 
#### Step 4: Cross-Validation
The Cross-Validation was performed on the Random Forest model. The training sample was randomly partitioned into 6 equal sized subsamples in order to estimate how accurately a predictive model would perform in practice. Of the 6 subsamples, a single subsample was retained as the validation data for testing the model, and the remaining k5 subsamples were used as training data. The cross-validation process then was repeated 6 times (the folds). 
### Results
Mean Absolute Error (MAE) was used to check the accuracy of the model. MAE was accounted to be 0.78, which is higher than expected 0.5. Hence more experimentation on how to treat the data should be done.


## 3. RF logreg Marketing
  This is a Python code with a model that predicts sing up for banking products given unbalanced data in 2017
## 4. Reservoir Engineering Proposal
  This is a technical document I wrote where I proposed how to integrate Data Science Operations for Reservor Egineering back in 2015 for Devon Energy
## 5. Projects Overview 
  This is a short pdf presentations of my previous work in oil and gas domain 
## 6. Spotfire Screenshot Examples
  This is a word file with screenshots of one of the dashboards that I built for Parsley Energy in 2016
