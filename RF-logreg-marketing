import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import sklearn as sk
from sklearn.model_selection import train_test_split
import scipy
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import itertools
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


#Read in the data from csv file
df = pd.read_csv("bank-additional-full.csv", sep=";")
df = df.dropna()

#Calculate basic stats and check out the data structure
#print(df.head)
#print(df.info())
#print(df.describe(include='all'))
"""
*****************************************************************************************
CHECK IF DATA IS BALANCED.
  Unbalanced data cannot go into a regression model without balancing the classes first.
*****************************************************************************************
"""
"""
print(df['y'].value_counts())
sns.countplot(x='y', data=df, palette='hls')
plt.show()

#calculate percentage of imbalance
count_no_sub = 36548
count_sub = 4640
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of subscription", pct_of_sub*100)
"""

"""
****************************************************************************
CATEGORICAL VARIABLES FREQUENCY PLOTS
****************************************************************************
"""
"""
sns.set(style='darkgrid')

print(df['job'].unique())
ax = sns.countplot(x='job', data=df)
pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')

print(df['marital'].unique())
ax = sns.countplot(x='marital', data=df)
pd.crosstab(df.marital,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Frequency of Purchase')

print(df['education'].unique())
ax = sns.countplot(x='education', data=df)
pd.crosstab(df.education,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Education Category')
plt.xlabel('Education Category')
plt.ylabel('Frequency of Purchase')

print(df['default'].unique())
ax = sns.countplot(x='default', data=df)
pd.crosstab(df.default,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Default Status')
plt.xlabel('Default Status')
plt.ylabel('Frequency of Purchase')

print(df['housing'].unique())
ax = sns.countplot(x='housing', data=df)
pd.crosstab(df.housing,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Housing Category')
plt.xlabel('Housing')
plt.ylabel('Frequency of Purchase')

print(df['loan'].unique())
ax = sns.countplot(x="loan", data=df)
pd.crosstab(df.loan,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Loan Category')
plt.xlabel('Loan')
plt.ylabel('Frequency of Purchase')

print(df['contact'].unique())
ax = sns.countplot(x='contact', data=df)
pd.crosstab(df.contact,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Contact Type')
plt.xlabel('Contact Type')
plt.ylabel('Frequency of Purchase')

print(df['month'].unique())
ax = sns.countplot(x='month', data=df)
pd.crosstab(df.month,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')

print(df['day_of_week'].unique())
ax = sns.countplot(x='day_of_week', data=df)
pd.crosstab(df.day_of_week,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Weekday')
plt.xlabel('Weekday')
plt.ylabel('Frequency of Purchase')

print(df['poutcome'].unique())
ax = sns.countplot(x='poutcome', data=df)
pd.crosstab(df.poutcome,df.y).plot(kind='bar')
plt.title('Purchase Frequency vs Outcome of the Previous Marketing Campaign')
plt.xlabel('Outcome of the Previous Marketing Campaign')
plt.ylabel('Frequency of Purchase')

plt.show()
"""
"""
****************************************************************************
CATEGORICAL VARIABLES GROUP BY TABLES
****************************************************************************
"""
"""
print(df.groupby('y').mean())
print(df.groupby('job').mean())
print(df.groupby('marital').mean())
print(df.groupby('education').mean())
print(df.groupby('default').mean())
print(df.groupby('housing').mean())
print(df.groupby('loan').mean())
print(df.groupby('contact').mean())
print(df.groupby('month').mean())
print(df.groupby('day_of_week').mean())
"""

"""
Interesting insights:average age for students is 25, aver age for retired is 62, aver age for management is 42. Divorced are older than married.
"""
"""
****************************************************************************
CATEGORICAL VARIABLES HISTOGRAMS
****************************************************************************
"""
"""
#sns.catplot(x="job", y="age",hue = "y", data=df, kind="violin");
#sns.catplot(x="month", y="duration",hue = "y", data=df, kind="violin");
sns.catplot(x="month", y="pdays",hue = "y", data=df, kind="violin");
#sns.catplot(x="month", y="previous",hue = "y", data=df, kind="violin");
#sns.catplot(x="marital", y="age",hue = "y", data=df, kind="violin");
plt.show()

"""
"""
****************************************************************************
NUMERICAL VARIABLES PLOT CORRELATION HEAT MAP
****************************************************************************
"""
"""
correlation = df.corr(method='pearson')
plt.figure(figsize=(25,10))
sns.heatmap(correlation, vmax=1, square=True,  annot=True )
plt.show()
"""
"""
****************************************************************************
NUMERICAL VARIABLES DENSITY PLOTS
****************************************************************************
"""
"""
sns.set(style='darkgrid')

ax = sns.distplot(df.y)
ax = sns.distplot(df.age)
ax = sns.distplot(df.duration)
ax = sns.distplot(df.campaign)
ax = sns.distplot(df.pdays)
ax = sns.distplot(df.previous)
ax = sns.distplot(df.euribor3m)
ax = sns.distplot(df['emp.var.rate'])
ax = sns.distplot(df['cons.price.idx'])
ax = sns.distplot(df['cons.conf.idx'])
ax = sns.distplot(df['nr.employed'])

plt.show()
"""
"""
****************************************************************************
DATA DROP
- Dropped 'duration' according to UCI site
- Dropped day and month
- Dropped contact method
- Dropped poutcome
****************************************************************************
"""
df.drop(['contact','day_of_week','month','duration','poutcome'], inplace=True, axis=1)

"""
****************************************************************************
ENCODE CATEGORICAL VALUES
- Done to insure numerical values as an input into ML model
****************************************************************************
"""
encoder = LabelEncoder()
# Label encoder
df['marital'] = encoder.fit_transform(df['marital'])
df['job'] = encoder.fit_transform(df['job'])
df['education'] = encoder.fit_transform(df['education'])
df['default'] = encoder.fit_transform(df['default'])
df['housing'] = encoder.fit_transform(df['housing'])
df['loan'] = encoder.fit_transform(df['loan'])
df['y'] = encoder.fit_transform(df['y'])

"""
****************************************************************************
NORMALIZE NUMERICAL VALUES
- Done to insure fair comparison
****************************************************************************
"""
min_max_scaler = preprocessing.MinMaxScaler()
data_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
#print(data_scaled)

"""
****************************************************************************
USE SMOTEEN TO ADRESS THE UNBALANCED DATA SET
- Combine over- and under-sampling using SMOTE and Edited Nearest Neighbours.
****************************************************************************
"""

smote_enn = SMOTEENN(random_state=0)
X = data_scaled.drop('y', axis=1)
y = data_scaled['y']
X_res, y_res = smote_enn.fit_sample(X, y)

"""
****************************************************************************
USE CROSSVALIDATION TO SPLIT DATA INTO TEST AND TRAIN
****************************************************************************
"""
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_res, y_res, test_size=0.3, random_state=0)

print("")
print("Train: ", len(X_train_resampled))
print("Test: ", len(X_test_resampled))

"""
****************************************************************************
RANDOM FOREST AND GRID SEARCH
****************************************************************************
"""


"""
clf = RandomForestClassifier(n_jobs=-1, random_state=7, max_features= 'sqrt', n_estimators=50)
clf.fit(X_train_resampled, y_train_resampled)

param_grid = {'n_estimators': [50, 500],'max_features': ['auto', 'sqrt', 'log2'],}

CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)

CV_clf.fit(X_train_resampled, y_train_resampled)
y_pred = clf.predict(X_test_resampled)
CV_clf.best_params_
"""


"""
****************************************************************************
REGRESSION AND GRID SEARCH
****************************************************************************
"""

logit_model=sm.Logit(y_train_resampled, X_train_resampled)
result=logit_model.fit()
print(result.summary2())

logreg = LogisticRegression()
logreg.fit(X_train_resampled, y_train_resampled)

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }

CV_logreg = GridSearchCV(estimator=logreg, param_grid=param_grid, cv= 5)

CV_logreg.fit(X_train_resampled, y_train_resampled)
y_pred = logreg.predict(X_test_resampled)
print(CV_logreg.best_params_)

"""
****************************************************************************
CHECK MODEL ACCURACY
****************************************************************************
"""



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

	# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_resampled,y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()

print("F1 Score: ", f1_score(y_test_resampled, y_pred, average="macro"))
print("Precision: ", precision_score(y_test_resampled, y_pred, average="macro"))
print("Recall: ", recall_score(y_test_resampled, y_pred, average="macro"))

"""
****************************************************************************
RECEIVER OPERATING CHARACTERISTIC
****************************************************************************
"""
roc_auc = roc_auc_score(y_test_resampled, y_pred)
fpr, tpr, thresholds = roc_curve(y_test_resampled,y_pred)
#roc_auc = auc(fpr,tpr)

# Plot ROC
plt.figure()
plt.plot(fpr, tpr, label='(area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
