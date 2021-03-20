import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from sklearn import preprocessing
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# ******************************************************************************************************
#                                 STEP 1: IMPORT DATA
# ******************************************************************************************************

train1 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/train/train_cohort_enrol_info.csv')
train2 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/train/train_sita_cohort_demo.csv')
train3 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/train/train_sita_cohort_dx.csv')
train4 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/train/train_sita_cohort_hcur.csv')
train5 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competitin/train/train_sita_cohort_rxproc.csv')

test1 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/test/test_cohort_enrol_info.csv')
test2 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/test/test_sita_cohort_demo.csv')
test3 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/test/test_sita_cohort_dx.csv')
test4 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/test/test_sita_cohort_hcur.csv')
test5 = pd.read_csv('C:/Users/anyul/OneDrive/Desktop/Competition/test/test_sita_cohort_rxproc.csv')

# Check how many columns and rows each data has.
# Check how first five rows of each data table looks like.

'''
print(train1.shape)
print(train1.head(10))
print(train2.shape)
print(train2.head(10))
print(train3.shape)
print(train3.head(10))
print(train4.shape)
print(train4.head(10))
print(train5.shape)
print(train5.head(10))
print(test1.shape)
print(test1.head(10))
print(test2.shape)
print(test2.head(10))
print(test3.shape)
print(test3.head(10))
print(test4.shape)
print(test4.head(10))
print(test5.shape)
print(test5.head(10))
'''
# ******************************************************************************************************
#                                  STEP 2: PROCESS TRAIN DATA
# ******************************************************************************************************


# STEP 2.1: Merge demographic data (cohort_enrol_infor.csv and Sita_cohort_demo.csv)
# Select only columns that are present in the research papers.
# Use left join on ENROLLID

resultdemo = pd.merge(train1[['ENROLID','EECLASS','EESTATU','EGEOLOC','EMPREL','ENRMON','HLTHPLAN','MEMDAYS','MHSACOVG','MSA','PHYFLAG','PLNTYP1','PLNTYP2','PLNTYP3','PLNTYP4','PLNTYP5','PLNTYP6','PLNTYP7','PLNTYP8','PLNTYP9','PLNTYP10','PLNTYP11','PLNTYP12','RX','WGTKEY']],train2[['ENROLID','AGE','SEX','REGION','URBAN','PLANTYP','sita_pdc_post2yr']], on='ENROLID').drop_duplicates(subset=['ENROLID'])

'''
# The column INDSTRY had mixed categorical and numeric values. Fix the values.
# print(resultdemo['INDSTRY'].unique())

resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==7.0,'7',resultdemo['INDSTRY'])
resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==6.0,'6',resultdemo['INDSTRY'])
resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==5.0,'5',resultdemo['INDSTRY'])
resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==4.0,'4',resultdemo['INDSTRY'])
resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==3.0,'3',resultdemo['INDSTRY'])
resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==2.0,'2',resultdemo['INDSTRY'])
resultdemo['INDSTRY']=np.where(resultdemo['INDSTRY']==1.0,'1',resultdemo['INDSTRY'])

# print(resultdemo['INDSTRY'].unique())
'''
'''
# Check for all null values
print(resultdemo.isnull().sum())
print(sum(resultdemo[resultdemo.INDSTRY.isnull()].index == resultdemo[resultdemo.INDSTRY.isnull()].index))
print(sum(resultdemo[resultdemo.MHSACOVG.isnull()].index == resultdemo[resultdemo.MHSACOVG.isnull()].index))
print(sum(resultdemo[resultdemo.MSA.isnull()].index == resultdemo[resultdemo.MSA.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP1.isnull()].index == resultdemo[resultdemo.PLNTYP1.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP2.isnull()].index == resultdemo[resultdemo.PLNTYP2.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP3.isnull()].index == resultdemo[resultdemo.PLNTYP3.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP4.isnull()].index == resultdemo[resultdemo.PLNTYP4.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP5.isnull()].index == resultdemo[resultdemo.PLNTYP5.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP6.isnull()].index == resultdemo[resultdemo.PLNTYP6.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP7.isnull()].index == resultdemo[resultdemo.PLNTYP7.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP8.isnull()].index == resultdemo[resultdemo.PLNTYP8.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP9.isnull()].index == resultdemo[resultdemo.PLNTYP9.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP10.isnull()].index == resultdemo[resultdemo.PLNTYP10.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP11.isnull()].index == resultdemo[resultdemo.PLNTYP11.isnull()].index))
print(sum(resultdemo[resultdemo.PLNTYP12.isnull()].index == resultdemo[resultdemo.PLNTYP12.isnull()].index))
print(sum(resultdemo[resultdemo.WGTKEY.isnull()].index == resultdemo[resultdemo.WGTKEY.isnull()].index))
print(sum(resultdemo[resultdemo.URBAN.isnull()].index == resultdemo[resultdemo.URBAN.isnull()].index))
'''

#Filter out all null values
resultdemo.dropna(inplace=True)

# STEP 2.2: Merge demographic data with comorbidity status and Charlson comorbidity index (CCI)
# Data tables: cohort_enrol_infor.csv, Sita_cohort_demo.csv and Sita_cohort_dx.csv
# Keep only psychiatric diseases and diabetes with/without complications based on the research
# Exclude the columns which are not present in Data Dictionary
# Use left join on ENROLLID

# print(resultdemo.head())
resultdemo_new = resultdemo.drop_duplicates()
# print(resultdemo_new.head())
resultdemo_new1 = resultdemo_new.drop_duplicates()
print(resultdemo_new1.head())

resultdemocom = pd.merge(resultdemo,train3[['ENROLID','YR1PRE_diabetes_with_complication','YR1PRE_diabetes_without_complication','YR1PRE_severe_psychotic_disorder','YR1PRE_type1_diabetes','YR1POST_depression','YR1POST_diabetes_with_complication','YR1POST_diabetes_without_complication','YR1POST_severe_psychotic_disorder','YR1POST_type1_diabetes','YR1PRE_alcoholism','YR1POST_alcoholism','YR1PRE_ketoacidosis_diabetes','YR1POST_secondary_diabetes','YR1PRE_secondary_diabetes']], on='ENROLID')
# print(resultdemocom.describe(include ='all'))

# STEP 2.3: Merge above data with healthcare resource utilization and cost data Sita_cohort_hcur.csv
# Data tables: cohort_enrol_infor.csv, Sita_cohort_demo.csv, Sita_cohort_dx.csv and
# The data columns are not available in Data Dictionary
# Previous research indicates that cost of the medication is correlated to the adherence rates
# Hence we find correlations using Spearman and Pearson coefficients to exclude columns.

# pairwise correlation
# print(train4.drop('ENROLID',axis=1).corr(method='spearman'))
# print('****************************************************')
# print(train4.drop('ENROLID',axis=1).corr(method='pearson'))

# Drop one column out of each pair with Spearman or Pearson higher than 0.6
new1_train4 = train4.drop('OOP_OUT_YR1PRE', axis=1)
new2_train4 = new1_train4.drop('HP_OUT_YR1PRE', axis=1)
new3_train4 = new2_train4.drop('OUTNUM_YR1PRE', axis=1)
new4_train4 = new3_train4.drop('OOP_OUT_YR1POST', axis=1)
new5_train4 = new4_train4.drop('HP_OUT_YR1POST', axis=1)
new6_train4 = new5_train4.drop('OUTNUM_YR1POST', axis=1)
new7_train4 = new6_train4.drop('HP_IN_YR1PRE', axis=1)
new8_train4 = new7_train4.drop('HP_IN_YR1POST', axis=1)
new9_train4 = new8_train4.drop('INCOST_T2MD_YR1POST', axis=1)
new10_train4 = new9_train4.drop('PHARCOST_YR1POST', axis=1)
new11_train4 = new10_train4.drop('OOP_PHAR_1YRPRE', axis=1)
new13_train4 = new11_train4.drop('HP_PHAR_YR1POST', axis=1)
new14_train4 = new13_train4.drop('PHARCOST_T2MD_YR1PRE', axis=1)
new15_train4 = new14_train4.drop('HP_T2MD_IN_YR1POST', axis=1)
new16_train4 = new15_train4.drop('HP_PHAR_YR1PRE', axis=1)
new17_train4 = new16_train4.drop('OOP_T2MD_IN_1YRPRE', axis=1)
new18_train4 = new17_train4.drop('OOP_T2MD_IN_1YRPOST', axis=1)
new19_train4 = new18_train4.drop('HP_T2MD_OUT_YR1POST', axis=1)
new20_train4 = new19_train4.drop('OUTCOST_T2MD_YR1PRE', axis=1)
new21_train4 = new20_train4.drop('OOP_T2MD_PHAR_1YRPOST', axis=1)
new22_train4 = new21_train4.drop('HP_T2MD_IN_YR1PRE', axis=1)
new23_train4 = new22_train4.drop('PHARCOST_T2MD_YR1POST', axis=1)
new_train4 = new23_train4.drop('HP_T2MD_PHAR_YR1PRE', axis=1)

# Note: can't seem to produce seaborn pairplot.
# Check the limit of variables for pairwise plot in Seaborn.

resultdemocomcost1 = pd.merge(resultdemocom,new_train4,on='ENROLID')
# print(resultdemocomcost1.describe(include='all'))
# print(resultdemocomcost1.isnull().sum())
# print(resultdemocomcost1.shape)

# drop all columns with null values larger than 60,000
resultdemocomcost2 = resultdemocomcost1.drop('INCOST_YR1PRE', axis=1)
resultdemocomcost3 = resultdemocomcost2.drop('INCOST_YR1POST', axis=1)
resultdemocomcost4 = resultdemocomcost3.drop('OOP_IN_YR1PRE', axis=1)
resultdemocomcost5 = resultdemocomcost4.drop('OOP_IN_YR1POST', axis=1)
resultdemocomcost6 = resultdemocomcost5.drop('INCOST_T2MD_YR1PRE', axis=1)
resultdemocomcost12 = resultdemocomcost6.drop('ERNUM_YR1PRE', axis=1)
resultdemocomcost13 = resultdemocomcost12.drop('ERNUM_YR1POST', axis=1)
resultdemocomcost14 = resultdemocomcost13.drop('HOSPNUM_YR1PRE', axis=1)
resultdemocomcost15 = resultdemocomcost14.drop('HOSPNUM_YR1POST', axis=1)
resultdemocomcost16 = resultdemocomcost15.drop('HOSPDAYS_YR1PRE', axis=1)
resultdemocomcost17 = resultdemocomcost16.drop('HOSPDAYS_YR1POST', axis=1)
resultdemocomcost18 = resultdemocomcost17.drop('HPDAYS_YR1PRE', axis=1)
resultdemocomcost = resultdemocomcost18.drop('HPDAYS_YR1POST', axis=1)

# drop all rows with missing values
resultdemocomcost.dropna(inplace=True)
# print(resultdemocomcost.isnull().sum())
# print(resultdemocomcost.shape)

# Merge above data with medical procedure data (table 5 in the Introduction document).

# Do not repeat the same mistake as with Table 4 and check for null values before examining correlations.
# print(train5.isnull().sum())
# Good news - results indicate that there are no missing values. We can proceed with correlations.
# print(train5.shape)

# The data columns are not available in a data dictionary.
# The data is a highly domain based knowledge and no previous research could be found.
# Hence we find correlations using Pearson and Spearman to see which columns to exclude before joining the data.

# pairwise correlation
# print(train5.drop('ENROLID',axis=1).corr(method='spearman'))
# print('****************************************************')
# print(train5.drop('ENROLID',axis=1).corr(method='pearson'))

# Drop one column out of each pair with high Spearman or Pearson coefficient
new1_train5 = train5.drop('beta.blocker_YR1POST', axis=1)
new2_train5 = new1_train5.drop('Bosentan_YR1POST', axis=1)
new3_train5 = new2_train5.drop('intestinal_obstruction_YR1POST', axis=1)
new4_train5 = new3_train5.drop('Disopyramide_YR1POST', axis=1)
new5_train5 = new4_train5.drop('Antidiabetic.Ag..SGLT.Inhibitr_YR1PRE', axis=1)
new6_train5 = new5_train5.drop('Gemfibrozil_YR1POST',axis=1)
new7_train5 = new6_train5.drop('Antidiabetic.Ag..SGLT.Inhibitr_YR1POST', axis=1)
new8_train5 = new7_train5.drop('hepatic_disease_YR1PRE', axis=1)
new9_train5 = new8_train5.drop('loop_diuretics_YR1POST', axis=1)
new10_train5 = new9_train5.drop('mao_inhibitor_YR1POST', axis=1)
new11_train5 = new10_train5.drop('warfarin_YR1POST', axis=1)
new_train5 = new11_train5.drop('carotid_revascualization_YR1PRE', axis=1)

# print(new12_train5.drop('ENROLID',axis=1).corr(method='spearman'))
# print('****************************************************')
# print(new12_train5.drop('ENROLID',axis=1).corr(method='pearson'))

resultdemocomcostmed = pd.merge(resultdemocomcost,new_train5,on='ENROLID')
# print(resultdemocomcostmed.describe(include='all'))
# print(resultdemocomcostmed.isnull().sum())
# print(resultdemocomcostmed.shape)

# Check it the dataset is unbalance
# Results indicate that the dataset is unbalanced
# print(resultdemocomcostmed['sita_pdc_post2yr'].value_counts())
sns.countplot(x='sita_pdc_post2yr',data=resultdemocomcostmed, palette='hls')
plt.show()
plt.savefig('count_plot')

# Check to make sure that data is cleaned and properly processed
# print(resultdemocomcostmed.info())
# print(resultdemocomcostmed.head())
# print(list(resultdemocomcostmed.columns))

# Use Label Encoder to convert all categorical data to numerical
number1 = LabelEncoder()
resultdemocomcostmed['INDSTRY']= number1.fit_transform(resultdemocomcostmed['INDSTRY'].astype('str'))
number2 = LabelEncoder()
resultdemocomcostmed['IDD']= number2.fit_transform(resultdemocomcostmed['IDD'].astype('str'))

# Save the target values in the original format for later
y_original = resultdemocomcostmed['sita_pdc_post2yr']

# Use Label Encoder to convert all floats to integers
number3 = LabelEncoder()
resultdemocomcostmed['MHSACOVG']= number3.fit_transform(resultdemocomcostmed['MHSACOVG'].astype('str'))
number4 = LabelEncoder()
resultdemocomcostmed['MSA']= number4.fit_transform(resultdemocomcostmed['MSA'].astype('str'))
number5 = LabelEncoder()
resultdemocomcostmed['PLNTYP1']= number5.fit_transform(resultdemocomcostmed['PLNTYP1'].astype('str'))
number6 = LabelEncoder()
resultdemocomcostmed['PLNTYP2']= number6.fit_transform(resultdemocomcostmed['PLNTYP2'].astype('str'))
number7 = LabelEncoder()
resultdemocomcostmed['PLNTYP3']= number7.fit_transform(resultdemocomcostmed['PLNTYP3'].astype('str'))
number8 = LabelEncoder()
resultdemocomcostmed['PLNTYP4']= number8.fit_transform(resultdemocomcostmed['PLNTYP4'].astype('str'))
number9 = LabelEncoder()
resultdemocomcostmed['PLNTYP5']= number9.fit_transform(resultdemocomcostmed['PLNTYP5'].astype('str'))
number10 = LabelEncoder()
resultdemocomcostmed['PLNTYP6']= number10.fit_transform(resultdemocomcostmed['PLNTYP6'].astype('str'))
number11 = LabelEncoder()
resultdemocomcostmed['PLNTYP7']= number11.fit_transform(resultdemocomcostmed['PLNTYP7'].astype('str'))
number12 = LabelEncoder()
resultdemocomcostmed['PLNTYP8']= number12.fit_transform(resultdemocomcostmed['PLNTYP8'].astype('str'))
number13 = LabelEncoder()
resultdemocomcostmed['PLNTYP9']= number13.fit_transform(resultdemocomcostmed['PLNTYP9'].astype('str'))
number14 = LabelEncoder()
resultdemocomcostmed['PLNTYP10']= number14.fit_transform(resultdemocomcostmed['PLNTYP10'].astype('str'))
number15 = LabelEncoder()
resultdemocomcostmed['PLNTYP11']= number15.fit_transform(resultdemocomcostmed['PLNTYP11'].astype('str'))
number16 = LabelEncoder()
resultdemocomcostmed['PLNTYP12']= number16.fit_transform(resultdemocomcostmed['PLNTYP12'].astype('str'))
number17 = LabelEncoder()
resultdemocomcostmed['WGTKEY']= number17.fit_transform(resultdemocomcostmed['WGTKEY'].astype('str'))
number18 = LabelEncoder()
resultdemocomcostmed['URBAN']= number18.fit_transform(resultdemocomcostmed['URBAN'].astype('str'))
number19 = LabelEncoder()
resultdemocomcostmed['sita_pdc_post2yr']= number19.fit_transform(resultdemocomcostmed['sita_pdc_post2yr'].astype('str'))
number20 = LabelEncoder()
resultdemocomcostmed['OUTCOST_YR1PRE']= number20.fit_transform(resultdemocomcostmed['OUTCOST_YR1PRE'].astype('str'))
number21 = LabelEncoder()
resultdemocomcostmed['OUTCOST_YR1POST']= number21.fit_transform(resultdemocomcostmed['OUTCOST_YR1POST'].astype('str'))
number22 = LabelEncoder()
resultdemocomcostmed['PHARCOST_YR1PRE']= number22.fit_transform(resultdemocomcostmed['PHARCOST_YR1PRE'].astype('str'))
number23 = LabelEncoder()
resultdemocomcostmed['OOP_PHAR_YR1POST']= number23.fit_transform(resultdemocomcostmed['OOP_PHAR_YR1POST'].astype('str'))
number24 = LabelEncoder()
resultdemocomcostmed['OUTCOST_T2MD_YR1POST']= number24.fit_transform(resultdemocomcostmed['OUTCOST_T2MD_YR1POST'].astype('str'))
number25 = LabelEncoder()
resultdemocomcostmed['OOP_T2MD_OUT_1YRPRE']= number25.fit_transform(resultdemocomcostmed['OOP_T2MD_OUT_1YRPRE'].astype('str'))
number26 = LabelEncoder()
resultdemocomcostmed['OOP_T2MD_OUT_1YRPOST']= number26.fit_transform(resultdemocomcostmed['OOP_T2MD_OUT_1YRPOST'].astype('str'))
number27 = LabelEncoder()
resultdemocomcostmed['OOP_T2MD_PHAR_1YRPRE']= number27.fit_transform(resultdemocomcostmed['OOP_T2MD_PHAR_1YRPRE'].astype('str'))
number28 = LabelEncoder()
resultdemocomcostmed['HP_T2MD_OUT_YR1PRE']= number28.fit_transform(resultdemocomcostmed['HP_T2MD_OUT_YR1PRE'].astype('str'))
number29 = LabelEncoder()
resultdemocomcostmed['HP_T2MD_PHAR_YR1POST']= number29.fit_transform(resultdemocomcostmed['HP_T2MD_PHAR_YR1POST'].astype('str'))

# print(resultdemocomcostmed.info())

# ******************************************************************************************************
#                                  STEP 3: PROCESS TEST DATA
# ******************************************************************************************************
# STEP 3.1: Dirty test data for cross validation later
# Hence we are only merging all test data tables into one and not doing anything else

# Merge all test tables
testdata1 = pd.merge(train1,train2,on='ENROLID')
testdata2 = pd.merge(testdata1,train3,on='ENROLID')
testdata3 = pd.merge(testdata2,train4,on='ENROLID')
testdata = pd.merge(testdata3,train5,on='ENROLID')


'''
print(testdata.describe(include='all'))
print(testdata.isnull().sum())
print(testdata.shape)
print(testdata.info())
print(testdata.head())
print(list(testdata.columns))
'''


# STEP 3.2: Merge demographic data (cohort_enrol_infor.csv and Sita_cohort_demo.csv)
# Select only columns that are present in the research papers.
# Use left join on ENROLLID

validationdemo = pd.merge(test1[['ENROLID','EECLASS','EESTATU','EGEOLOC','EMPREL','ENRMON','HLTHPLAN','INDSTRY','MEMDAYS','MHSACOVG','MSA','PHYFLAG','PLNTYP1','PLNTYP2','PLNTYP3','PLNTYP4','PLNTYP5','PLNTYP6','PLNTYP7','PLNTYP8','PLNTYP9','PLNTYP10','PLNTYP11','PLNTYP12','RX','WGTKEY']],test2[['ENROLID','AGE','SEX','REGION','URBAN','PLANTYP']], on='ENROLID')

# The column INDSTRY had mixed categorical and numeric values. Fix the values.
# print(validationdemo['INDSTRY'].unique())

validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==7.0,'7',validationdemo['INDSTRY'])
validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==6.0,'6',validationdemo['INDSTRY'])
validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==5.0,'5',validationdemo['INDSTRY'])
validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==4.0,'4',validationdemo['INDSTRY'])
validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==3.0,'3',validationdemo['INDSTRY'])
validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==2.0,'2',validationdemo['INDSTRY'])
validationdemo['INDSTRY']=np.where(validationdemo['INDSTRY']==1.0,'1',validationdemo['INDSTRY'])

# print(validationdemo['INDSTRY'].unique())
'''
# Check for all null values
print(validationdemo.isnull().sum())
print(sum(validationdemo[validationdemo.INDSTRY.isnull()].index == validationdemo[validationdemo.INDSTRY.isnull()].index))
print(sum(validationdemo[validationdemo.MHSACOVG.isnull()].index == validationdemo[validationdemo.MHSACOVG.isnull()].index))
print(sum(validationdemo[validationdemo.MSA.isnull()].index == validationdemo[validationdemo.MSA.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP1.isnull()].index == validationdemo[validationdemo.PLNTYP1.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP2.isnull()].index == validationdemo[validationdemo.PLNTYP2.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP3.isnull()].index == validationdemo[validationdemo.PLNTYP3.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP4.isnull()].index == validationdemo[validationdemo.PLNTYP4.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP5.isnull()].index == validationdemo[validationdemo.PLNTYP5.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP6.isnull()].index == validationdemo[validationdemo.PLNTYP6.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP7.isnull()].index == validationdemo[validationdemo.PLNTYP7.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP8.isnull()].index == validationdemo[validationdemo.PLNTYP8.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP9.isnull()].index == validationdemo[validationdemo.PLNTYP9.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP10.isnull()].index == validationdemo[validationdemo.PLNTYP10.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP11.isnull()].index == validationdemo[validationdemo.PLNTYP11.isnull()].index))
print(sum(validationdemo[validationdemo.PLNTYP12.isnull()].index == validationdemo[validationdemo.PLNTYP12.isnull()].index))
print(sum(validationdemo[validationdemo.WGTKEY.isnull()].index == validationdemo[validationdemo.WGTKEY.isnull()].index))
print(sum(validationdemo[validationdemo.URBAN.isnull()].index == validationdemo[validationdemo.URBAN.isnull()].index))
'''

#Filter out all null values
validationdemo.dropna(inplace=True)

# STEP 3.3: Merge demographic data with comorbidity status and Charlson comorbidity index (CCI)
# Data tables: cohort_enrol_infor.csv, Sita_cohort_demo.csv and Sita_cohort_dx.csv
# Keep only psychiatric diseases and diabetes with/without complications based on the research
# Exclude the columns which are not present in Data Dictionary
# Use left join on ENROLLID

validationdemocom = pd.merge(validationdemo,test3[['ENROLID','YR1PRE_diabetes_with_complication','YR1PRE_diabetes_without_complication','YR1PRE_severe_psychotic_disorder','YR1PRE_type1_diabetes','YR1POST_depression','YR1POST_diabetes_with_complication','YR1POST_diabetes_without_complication','YR1POST_severe_psychotic_disorder','YR1POST_type1_diabetes','YR1PRE_alcoholism','YR1POST_alcoholism','YR1PRE_ketoacidosis_diabetes','YR1POST_secondary_diabetes','YR1PRE_secondary_diabetes']], on='ENROLID')
# print(validationdemocom.describe(include ='all'))

# STEP 3.4: Merge above data with healthcare resource utilization and cost data Sita_cohort_hcur.csv
# Data tables: cohort_enrol_infor.csv, Sita_cohort_demo.csv, Sita_cohort_dx.csv and
# The data columns are not available in Data Dictionary
# Previous research indicates that cost of the medication is correlated to the adherence rates
# Hence we find correlations using Spearman and Pearson coefficients to exclude columns.

# pairwise correlation
# print(test4.drop('ENROLID',axis=1).corr(method='spearman'))
# print('****************************************************')
# print(test4.drop('ENROLID',axis=1).corr(method='pearson'))

# Drop one column out of each pair with Spearman or Pearson higher than 0.6
new1_test4 = test4.drop('OOP_OUT_YR1PRE', axis=1)
new2_test4 = new1_test4.drop('HP_OUT_YR1PRE', axis=1)
new3_test4 = new2_test4.drop('OUTNUM_YR1PRE', axis=1)
new4_test4 = new3_test4.drop('OOP_OUT_YR1POST', axis=1)
new5_test4 = new4_test4.drop('HP_OUT_YR1POST', axis=1)
new6_test4 = new5_test4.drop('OUTNUM_YR1POST', axis=1)
new7_test4 = new6_test4.drop('HP_IN_YR1PRE', axis=1)
new8_test4 = new7_test4.drop('HP_IN_YR1POST', axis=1)
new9_test4 = new8_test4.drop('INCOST_T2MD_YR1POST', axis=1)
new10_test4 = new9_test4.drop('PHARCOST_YR1POST', axis=1)
new11_test4 = new10_test4.drop('OOP_PHAR_1YRPRE', axis=1)
new13_test4 = new11_test4.drop('HP_PHAR_YR1POST', axis=1)
new14_test4 = new13_test4.drop('PHARCOST_T2MD_YR1PRE', axis=1)
new15_test4 = new14_test4.drop('HP_T2MD_IN_YR1POST', axis=1)
new16_test4 = new15_test4.drop('HP_PHAR_YR1PRE', axis=1)
new17_test4 = new16_test4.drop('OOP_T2MD_IN_1YRPRE', axis=1)
new18_test4 = new17_test4.drop('OOP_T2MD_IN_1YRPOST', axis=1)
new19_test4 = new18_test4.drop('HP_T2MD_OUT_YR1POST', axis=1)
new20_test4 = new19_test4.drop('OUTCOST_T2MD_YR1PRE', axis=1)
new21_test4 = new20_test4.drop('OOP_T2MD_PHAR_1YRPOST', axis=1)
new22_test4 = new21_test4.drop('HP_T2MD_IN_YR1PRE', axis=1)
new23_test4 = new22_test4.drop('PHARCOST_T2MD_YR1POST', axis=1)
new_test4 = new23_test4.drop('HP_T2MD_PHAR_YR1PRE', axis=1)

# Note: can't seem to produce seaborn pairplot.
# Check the limit of variables for pairwise plot in Seaborn.

validationdemocomcost1 = pd.merge(validationdemocom,new_test4,on='ENROLID')
# print(validationdemocomcost1.describe(include='all'))
# print(validationdemocomcost1.isnull().sum())
# print(validationdemocomcost1.shape)

# drop all columns with null values larger than 60,000
validationdemocomcost2 = validationdemocomcost1.drop('INCOST_YR1PRE', axis=1)
validationdemocomcost3 = validationdemocomcost2.drop('INCOST_YR1POST', axis=1)
validationdemocomcost4 = validationdemocomcost3.drop('OOP_IN_YR1PRE', axis=1)
validationdemocomcost5 = validationdemocomcost4.drop('OOP_IN_YR1POST', axis=1)
validationdemocomcost6 = validationdemocomcost5.drop('INCOST_T2MD_YR1PRE', axis=1)
validationdemocomcost12 = validationdemocomcost6.drop('ERNUM_YR1PRE', axis=1)
validationdemocomcost13 = validationdemocomcost12.drop('ERNUM_YR1POST', axis=1)
validationdemocomcost14 = validationdemocomcost13.drop('HOSPNUM_YR1PRE', axis=1)
validationdemocomcost15 = validationdemocomcost14.drop('HOSPNUM_YR1POST', axis=1)
validationdemocomcost16 = validationdemocomcost15.drop('HOSPDAYS_YR1PRE', axis=1)
validationdemocomcost17 = validationdemocomcost16.drop('HOSPDAYS_YR1POST', axis=1)
validationdemocomcost18 = validationdemocomcost17.drop('HPDAYS_YR1PRE', axis=1)
validationdemocomcost = validationdemocomcost18.drop('HPDAYS_YR1POST', axis=1)

# drop all rows with missing values
validationdemocomcost.dropna(inplace=True)
# print(validationdemocomcost.isnull().sum())
# print(validationdemocomcost.shape)

# Merge above data with medical procedure data (table 5 in the Introduction document).

# Do not repeat the same mistake as with Table 4 and check for null values before examining correlations.
# print(test5.isnull().sum())
# Good news - results indicate that there are no missing values. We can proceed with correlations.
# print(test5.shape)

# The data columns are not available in a data dictionary.
# The data is a highly domain based knowledge and no previous research could be found.
# Hence we find correlations using Pearson and Spearman to see which columns to exclude before joining the data.

# pairwise correlation
# print(test5.drop('ENROLID',axis=1).corr(method='spearman'))
# print('****************************************************')
# print(test5.drop('ENROLID',axis=1).corr(method='pearson'))

# Drop one column out of each pair with high Spearman or Pearson coefficient
new1_test5 = test5.drop('beta.blocker_YR1POST', axis=1)
new2_test5 = new1_test5.drop('Bosentan_YR1POST', axis=1)
new3_test5 = new2_test5.drop('intestinal_obstruction_YR1POST', axis=1)
new4_test5 = new3_test5.drop('Disopyramide_YR1POST', axis=1)
new5_test5 = new4_test5.drop('Antidiabetic.Ag..SGLT.Inhibitr_YR1PRE', axis=1)
new6_test5 = new5_test5.drop('Gemfibrozil_YR1POST',axis=1)
new7_test5 = new6_test5.drop('Antidiabetic.Ag..SGLT.Inhibitr_YR1POST', axis=1)
new8_test5 = new7_test5.drop('hepatic_disease_YR1PRE', axis=1)
new9_test5 = new8_test5.drop('loop_diuretics_YR1POST', axis=1)
new10_test5 = new9_test5.drop('mao_inhibitor_YR1POST', axis=1)
new11_test5 = new10_test5.drop('warfarin_YR1POST', axis=1)
new_test5 = new11_test5.drop('carotid_revascualization_YR1PRE', axis=1)

# print(new12_test5.drop('ENROLID',axis=1).corr(method='spearman'))
# print('****************************************************')
# print(new12_test5.drop('ENROLID',axis=1).corr(method='pearson'))

validationdemocomcostmed = pd.merge(validationdemocomcost,new_test5,on='ENROLID')
# print(validationdemocomcostmed.describe(include='all'))
# print(validationdemocomcostmed.isnull().sum())
# print(validationdemocomcostmed.shape)

# Check to make sure that data is cleaned and properly processed
# print(validationdemocomcostmed.info())
# print(validationdemocomcostmed.head())
# print(list(validationdemocomcostmed.columns))

# Use Label Encoder to convert all categorical data to numerical
number1 = LabelEncoder()
validationdemocomcostmed['INDSTRY']= number1.fit_transform(validationdemocomcostmed['INDSTRY'].astype('str'))
# print(validationdemocomcostmed['INDSTRY'].unique())
number2 = LabelEncoder()
validationdemocomcostmed['IDD']= number2.fit_transform(validationdemocomcostmed['IDD'].astype('str'))
# print(validationdemocomcostmed['IDD'].unique())
# print(validationdemocomcostmed.info())

# Use Label Encoder to convert all floats to integers
number3 = LabelEncoder()
validationdemocomcostmed['MHSACOVG']= number3.fit_transform(validationdemocomcostmed['MHSACOVG'].astype('str'))
number4 = LabelEncoder()
validationdemocomcostmed['MSA']= number4.fit_transform(validationdemocomcostmed['MSA'].astype('str'))
number5 = LabelEncoder()
validationdemocomcostmed['PLNTYP1']= number5.fit_transform(validationdemocomcostmed['PLNTYP1'].astype('str'))
number6 = LabelEncoder()
validationdemocomcostmed['PLNTYP2']= number6.fit_transform(validationdemocomcostmed['PLNTYP2'].astype('str'))
number7 = LabelEncoder()
validationdemocomcostmed['PLNTYP3']= number7.fit_transform(validationdemocomcostmed['PLNTYP3'].astype('str'))
number8 = LabelEncoder()
validationdemocomcostmed['PLNTYP4']= number8.fit_transform(validationdemocomcostmed['PLNTYP4'].astype('str'))
number9 = LabelEncoder()
validationdemocomcostmed['PLNTYP5']= number9.fit_transform(validationdemocomcostmed['PLNTYP5'].astype('str'))
number10 = LabelEncoder()
validationdemocomcostmed['PLNTYP6']= number10.fit_transform(validationdemocomcostmed['PLNTYP6'].astype('str'))
number11 = LabelEncoder()
validationdemocomcostmed['PLNTYP7']= number11.fit_transform(validationdemocomcostmed['PLNTYP7'].astype('str'))
number12 = LabelEncoder()
validationdemocomcostmed['PLNTYP8']= number12.fit_transform(validationdemocomcostmed['PLNTYP8'].astype('str'))
number13 = LabelEncoder()
validationdemocomcostmed['PLNTYP9']= number13.fit_transform(validationdemocomcostmed['PLNTYP9'].astype('str'))
number14 = LabelEncoder()
validationdemocomcostmed['PLNTYP10']= number14.fit_transform(validationdemocomcostmed['PLNTYP10'].astype('str'))
number15 = LabelEncoder()
validationdemocomcostmed['PLNTYP11']= number15.fit_transform(validationdemocomcostmed['PLNTYP11'].astype('str'))
number16 = LabelEncoder()
validationdemocomcostmed['PLNTYP12']= number16.fit_transform(validationdemocomcostmed['PLNTYP12'].astype('str'))
number17 = LabelEncoder()
validationdemocomcostmed['WGTKEY']= number17.fit_transform(validationdemocomcostmed['WGTKEY'].astype('str'))
number18 = LabelEncoder()
validationdemocomcostmed['URBAN']= number18.fit_transform(validationdemocomcostmed['URBAN'].astype('str'))
number20 = LabelEncoder()
validationdemocomcostmed['OUTCOST_YR1PRE']= number20.fit_transform(validationdemocomcostmed['OUTCOST_YR1PRE'].astype('str'))
number21 = LabelEncoder()
validationdemocomcostmed['OUTCOST_YR1POST']= number21.fit_transform(validationdemocomcostmed['OUTCOST_YR1POST'].astype('str'))
number22 = LabelEncoder()
validationdemocomcostmed['PHARCOST_YR1PRE']= number22.fit_transform(validationdemocomcostmed['PHARCOST_YR1PRE'].astype('str'))
number23 = LabelEncoder()
validationdemocomcostmed['OOP_PHAR_YR1POST']= number23.fit_transform(validationdemocomcostmed['OOP_PHAR_YR1POST'].astype('str'))
number24 = LabelEncoder()
validationdemocomcostmed['OUTCOST_T2MD_YR1POST']= number24.fit_transform(validationdemocomcostmed['OUTCOST_T2MD_YR1POST'].astype('str'))
number25 = LabelEncoder()
validationdemocomcostmed['OOP_T2MD_OUT_1YRPRE']= number25.fit_transform(validationdemocomcostmed['OOP_T2MD_OUT_1YRPRE'].astype('str'))
number26 = LabelEncoder()
validationdemocomcostmed['OOP_T2MD_OUT_1YRPOST']= number26.fit_transform(validationdemocomcostmed['OOP_T2MD_OUT_1YRPOST'].astype('str'))
number27 = LabelEncoder()
validationdemocomcostmed['OOP_T2MD_PHAR_1YRPRE']= number27.fit_transform(validationdemocomcostmed['OOP_T2MD_PHAR_1YRPRE'].astype('str'))
number28 = LabelEncoder()
validationdemocomcostmed['HP_T2MD_OUT_YR1PRE']= number28.fit_transform(validationdemocomcostmed['HP_T2MD_OUT_YR1PRE'].astype('str'))
number29 = LabelEncoder()
validationdemocomcostmed['HP_T2MD_PHAR_YR1POST']= number29.fit_transform(validationdemocomcostmed['HP_T2MD_PHAR_YR1POST'].astype('str'))

# ******************************************************************************************************
#                                  STEP 4: MODELLING
# ******************************************************************************************************
# STEP 4: Train model on imbalanced data
# Separate input features (X) and target variable (y)
y = resultdemocomcostmed['sita_pdc_post2yr']
X1 = resultdemocomcostmed.drop('sita_pdc_post2yr', axis=1)
X2 = X1.drop('ENROLID', axis=1)
X = X2.drop('IDD', axis=1)

'''
# Split given test data for cross-validation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)
print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)

clf0 = RandomForestClassifier()
clf0.fit(X_train, y_train)
print(clf0.score(X_valid,y_valid))
'''

# Load the test data to predict the target for competition
X_test1 = validationdemocomcostmed.drop('ENROLID',axis = 1)
X_test = X_test1.drop('IDD',axis = 1)

clf = RandomForestClassifier()
clf.fit(X, y)
pred_y = clf.predict(X)
print(accuracy_score(pred_y,y))
print(clf.score(X,y))
# print(clf0.score(X,y))
# print(np.unique( pred_y))

pred_y_test = clf.predict(X_test)
# pred_y_test1 = clf0.predict(X_test)
# print(np.unique(pred_y_test))
true_y = number19.inverse_transform(pred_y_test)
# pred_true_y = number19.inverse_transform(pred_y)
# print(np.unique(true_y))
'''
clf1 = RandomForestRegressor()
clf1.fit(X,y)
pred_y1 = clf1.predict(X)
print(accuracy_score(pred_y,y))
print(clf1.score(X,y))

pred_y_test1 = clf1.predict(X_test)
# pred_y_test1 = clf0.predict(X_test)
# print(np.unique(pred_y_test))
true_y1 = number19.inverse_transform(pred_y_test1)
# print(np.unique(true_y))

'''
# ******************************************************************************************************
#                                  STEP 5:  CALCULATE MAE
# ******************************************************************************************************


residual = np.array(y) - np.array(pred_y)
MAE = np.mean(abs(residual))
print(mean_absolute_error(y,pred_y))
print(mean_absolute_error(y_original,true_y))



# ******************************************************************************************************
#                                  STEP 6:  GENERATE FILE FOR SUBMISSION
# ******************************************************************************************************
#submission = pd.DataFrame({'ID':validationdemocomcostmed['ENROLID'],'prediction':true_y})
#print(submission.head())

