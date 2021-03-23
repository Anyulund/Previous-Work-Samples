# Previous Work Samples

Welcome! 

This folder contains samples of some work I did for previous companies to demonstrate the following skills: Python coding, data cleaning and preprocessing, SQL coding, data modeling,  buidling dashboards, writing technical documents and doing research. All these projects are simple.Some of them were done at the beginning of my career and need better commenting,refactoring and clean up. This is work in progress. Please let me know if you have any questions, suggestions or inquiries for more complex projects. 

If you want to see my NLP work, please go to my NLP folder. 

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
* Health literacy
* Literacy
* Treatment cost 
* Age
* Ethnicity
* Prescription fill rates (Medication Possession Ratio)
* Course completion
* Health and disease management
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
  This is a Python code was an experimentation model on data related with direct marketing campaigns (phone calls) of a banking institution in 2017. The classification goal was to predict if the client will subscribe a term deposit.The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.
### Data used: 
* Age (numeric)
* Job : type of job (categorical: 'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown')
* Marital : marital status (categorical: 'divorced', 'married', 'single', 'unknown' ; note: 'divorced' means divorced or widowed)
* Education (categorical: 'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown')
* Default: has credit in default? (categorical: 'no', 'yes', 'unknown')
* Housing: has housing loan? (categorical: 'no', 'yes', 'unknown')
* Loan: has personal loan? (categorical: 'no', 'yes', 'unknown')
* Contact: contact communication type (categorical: 'cellular','telephone')
* Month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
* Day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
* Duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
* Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
* Pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
* Previous: number of contacts performed before this campaign and for this client (numeric)
* Poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
* Emp.var.rate: employment variation rate - quarterly indicator (numeric)
* Cons.price.idx: consumer price index - monthly indicator (numeric)
* Cons.conf.idx: consumer confidence index - monthly indicator (numeric)
* Euribor3m: euribor 3 month rate - daily indicator (numeric)
* Nr.employed: number of employees - quarterly indicator (numeric)
  
  
## 4. Reservoir Engineering Proposal
  This is a technical document I wrote where I proposed how to integrate Data Science Operations for Reservor Egineering back in 2015 for Devon Energy
## 5. Projects Overview 
  This is a short pdf presentations of my previous work in oil and gas domain 
## 6. Spotfire Screenshot Examples
  This is a word file with screenshots of one of the dashboards that I built for Parsley Energy in 2016
