
# Import Packages
from collections import OrderedDict
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandasql as pdsql
import sys
from sklearn.metrics import auc


# Define Functions
def f_bins(row):
    if row['MODEL_SCORE_NR'] == 0:
        val, label = 0, "0 (Score 0)"
   elif row['MODEL_SCORE_NR'] >= 1 and row['MODEL_SCORE_NR'] <= 49:
        val, label = 1, "1 (Scores 1-49)"
   elif row['MODEL_SCORE_NR'] >= 50 and row['MODEL_SCORE_NR'] <= 99:
        val, label = 2, "2 (Scores 50-99)"
   elif row['MODEL_SCORE_NR'] >= 100 and row['MODEL_SCORE_NR'] <= 149:
        val, label = 3, "3 (Scores 100-149)"
   elif row['MODEL_SCORE_NR'] >= 150 and row['MODEL_SCORE_NR'] <= 199:
        val, label = 4, "4 (Scores 150-199)"
   elif row['MODEL_SCORE_NR'] >= 200 and row['MODEL_SCORE_NR'] <= 249:
        val, label = 5, "5 (Scores 200-249)"
   elif row['MODEL_SCORE_NR'] >= 250 and row['MODEL_SCORE_NR'] <= 299:
        val, label = 6, "6 (Scores 250-299)"
   elif row['MODEL_SCORE_NR'] >= 300 and row['MODEL_SCORE_NR'] <= 349:
        val, label = 7, "7 (Scores 300-349)"
   elif row['MODEL_SCORE_NR'] >= 350 and row['MODEL_SCORE_NR'] <= 399:
        val, label = 8, "8 (Scores 350-399)"
   elif row['MODEL_SCORE_NR'] >= 400 and row['MODEL_SCORE_NR'] <= 449:
        val, label = 9, "9 (Scores 400-449)"
   elif row['MODEL_SCORE_NR'] >= 450 and row['MODEL_SCORE_NR'] <= 499:
        val, label = 10, "10 (Scores 450-499)"
   elif row['MODEL_SCORE_NR'] >= 500 and row['MODEL_SCORE_NR'] <= 549:
        val, label = 11, "11 (Scores 500-549)"
   elif row['MODEL_SCORE_NR'] >= 550 and row['MODEL_SCORE_NR'] <= 599:
        val, label = 12, "12 (Scores 550-599)"
   elif row['MODEL_SCORE_NR'] >= 600 and row['MODEL_SCORE_NR'] <= 649:
        val, label = 13, "13 (Scores 600-649)"
   elif row['MODEL_SCORE_NR'] >= 650 and row['MODEL_SCORE_NR'] <= 699:
        val, label = 14, "14 (Scores 650-699)"
   elif row['MODEL_SCORE_NR'] >= 700 and row['MODEL_SCORE_NR'] <= 749:
        val, label = 15, "15 (Scores 700-749)"
   elif row['MODEL_SCORE_NR'] >= 750 and row['MODEL_SCORE_NR'] <= 799:
        val, label = 16, "16 (Scores 750-799)"
   elif row['MODEL_SCORE_NR'] >= 800 and row['MODEL_SCORE_NR'] <= 849:
        val, label = 17, "17 (Scores 800-849)"
   elif row['MODEL_SCORE_NR'] >= 850 and row['MODEL_SCORE_NR'] <= 899:
        val, label = 18, "18 (Scores 850-899)"
   elif row['MODEL_SCORE_NR'] >= 900 and row['MODEL_SCORE_NR'] <= 949:
        val, label = 19, "19 (Scores 900-949)"
   elif row['MODEL_SCORE_NR'] >= 950:
        val, label = 20, "20 (Scores 950-999)"
   else:
        val, label = -1, "ERROR"
   # return val, label
   return pd.Series([val, label])

def f_nonzero(row):
    if row['MODEL_SCORE_NR'] == 0:
        val = 0
   elif row['MODEL_SCORE_NR'] >= 1:
        val = 1
   else:
        val = -1
   return val


def data_prep(group, out_directory, out_model, out_str):
   
    temp_df = group
 
    temp_df = temp_df.sort_values('MODEL_SCORE_NR', ascending=False)
    # temp_df = group.sort_values('MODEL_SCORE_NR', ascending=False)

   temp_df[['CUMSUM_NUM_TXNS']] = temp_df[['SUM_NUM_TXNS']].cumsum(axis=0)
    temp_df[['CUMSUM_NUM_TXNS_VALID']] = temp_df[['SUM_NUM_TXNS_VALID']].cumsum(axis=0)
    temp_df[['CUMSUM_NUM_TXNS_FRAUD']] = temp_df[['SUM_NUM_TXNS_FRAUD']].cumsum(axis=0)

    temp_df[['CUMSUM_APPROVED_TRANS_VALID']] = temp_df[['SUM_FRAUD_DECLINE_TRANS_VALID']].cumsum(axis=0)
    temp_df[['CUMSUM_FRAUD_DECLINE_TRANS_VALID']] = temp_df[['SUM_FRAUD_DECLINE_TRANS_VALID']].cumsum(axis=0)
    temp_df[['CUMSUM_NONFRAUD_DECLINE_TRANS_VALID']] = temp_df[['SUM_NONFRAUD_DECLINE_TRANS_VALID']].cumsum(axis=0)

    temp_df[['CUMSUM_APPROVED_TRANS_FRAUD']] = temp_df[['SUM_APPROVED_TRANS_FRAUD']].cumsum(axis=0)
    temp_df[['CUMSUM_FRAUD_DECLINE_TRANS_FRAUD']] = temp_df[['SUM_FRAUD_DECLINE_TRANS_FRAUD']].cumsum(axis=0)
    temp_df[['CUMSUM_NONFRAUD_DECLINE_TRANS_FRAUD']] = temp_df[['SUM_NONFRAUD_DECLINE_TRANS_FRAUD']].cumsum(axis=0)

    temp_df = temp_df.sort_values('MODEL_SCORE_NR', ascending=True)

    total_valid, total_fraud = temp_df[['SUM_NUM_TXNS_VALID','SUM_NUM_TXNS_FRAUD']].sum(axis=0)

    temp_df['CUMSUM_NUM_TXNS_VALID_DIST'] = temp_df['CUMSUM_NUM_TXNS_VALID']/total_valid
    temp_df['CUMSUM_NUM_TXNS_FRAUD_DIST'] = temp_df['CUMSUM_NUM_TXNS_FRAUD']/total_fraud

    temp_df.to_csv(out_directory + out_model + "_data_" +  out_str  + ".csv")

    return temp_df


def calc_ks(group, out_directory, out_model, out_str):

    temp_df = group

    if out_str == 'monthly':
       temp_year = list(set(temp_df['YEAR'].tolist()))[0]
       temp_month = list(set(temp_df['MONTH'].tolist()))[0]
       temp_period = str(temp_year) + '_' + str(temp_month).zfill(2)
    else:
        temp_year = 'Overall'
       temp_month = 'Overall'
       temp_period = 'Overall'

   temp_df['KS_DIST'] = temp_df['CUMSUM_NUM_TXNS_FRAUD_DIST'] - temp_df['CUMSUM_NUM_TXNS_VALID_DIST']

    ks = float(temp_df['KS_DIST'].max())
    ks_scaled = ks * 100

   ks_max_base = float(temp_df.loc[temp_df['KS_DIST'].idxmax(),'MODEL_SCORE_NR_SCALE_INV'])
    ks_max_target = float(temp_df.loc[temp_df['KS_DIST'].idxmax(),'CUMSUM_NUM_TXNS_FRAUD_DIST'])
    ks_max_nontarget = float(temp_df.loc[temp_df['KS_DIST'].idxmax(),'CUMSUM_NUM_TXNS_VALID_DIST'])

    plt.plot(temp_df['MODEL_SCORE_NR_SCALE_INV'],
             temp_df['CUMSUM_NUM_TXNS_VALID_DIST'],
             zorder=1, linestyle='-', color='blue',
             linewidth=3, label='Cumulative Dist Nontarget')
    plt.plot(temp_df['MODEL_SCORE_NR_SCALE_INV'],
             temp_df['CUMSUM_NUM_TXNS_FRAUD_DIST'],
             zorder=2, linestyle='-', color='red',
             linewidth=3, label='Cumulative Dist Target')
    plt.plot(temp_df['MODEL_SCORE_NR_SCALE_INV'].tolist() + [1.1],
             temp_df['MODEL_SCORE_NR_SCALE_INV'].tolist() + [1.1],
             zorder=3, linestyle='-', color='grey', linewidth=3, label='Baseline')
    plt.plot([ks_max_base, ks_max_base], [ks_max_target, ks_max_nontarget],
             zorder=4, linestyle='--', color='black',
             linewidth=3, label='K-S (' + str(ks_scaled) + ')')
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.title('K-S Statistic')
    plt.legend(loc='upper left')
    plt.xlabel('Score')
    plt.ylabel('Distribution Percentage')
    plt.savefig(out_directory + out_model + '_ks_' + temp_period + '.png')
    plt.show()
    plt.close()

    print(str(temp_period) + ' KS: ' + str(ks_scaled))

    return ks_scaled


def calc_roc(group, out_directory, out_model, out_str):

    temp_df = group

    if out_str == 'monthly':
       temp_year = list(set(temp_df['YEAR'].tolist()))[0]
       temp_month = list(set(temp_df['MONTH'].tolist()))[0]
       temp_period = str(temp_year) + '_' + str(temp_month).zfill(2)
    else:
        temp_period = 'Overall'

   temp_df = temp_df.sort_values('MODEL_SCORE_NR', ascending=True)

    total_valid, total_fraud = temp_df[['SUM_NUM_TXNS_VALID','SUM_NUM_TXNS_FRAUD']].sum(axis=0)

    temp_scores = temp_df['MODEL_SCORE_NR'].tolist()

    tp = []
    fp = []
    tn = []
    fn = []

    tpr = []
    tnr = []
    fnr = []
    fpr = []
    ppv = []
    npv = []
    fdr = []
    fort= []
    acc = []


    out_handle = open(out_directory + out_model + '_stats_' + temp_period + '.csv','w')

    header_str = 'score,fp,tp,fn,tn,fpr_baseline,tpr_baseline,fpr,tpr,tnr,fnr,ppv,npv,fdr,for,acc'

   out_handle.write(header_str + '\n')

    fpr_all = []
    tpr_all = []

    for idx, score in enumerate(temp_scores, start=1):

        fpr_baseline = score/1000
       tpr_baseline = score/1000
       fp_temp = int(temp_df.loc[(temp_df["MODEL_SCORE_NR"] == score),'CUMSUM_NUM_TXNS_VALID'])
        tp_temp = int(temp_df.loc[(temp_df["MODEL_SCORE_NR"] == score),'CUMSUM_NUM_TXNS_FRAUD'])
        tn_temp = int(total_valid - fp_temp)
        fn_temp = int(total_fraud - tp_temp)

        try:
            tpr_temp = tp_temp / (tp_temp + fn_temp)
        except ZeroDivisionError:
            tpr_temp = 'NA'
       try:
            tnr_temp = tn_temp / (tn_temp + fp_temp)
        except ZeroDivisionError:
            tnr_temp = 'NA'
       try:
            fnr_temp = fn_temp / (tp_temp + fn_temp)
        except ZeroDivisionError:
            fnr_temp = 'NA'
       try:
            fpr_temp = fp_temp / (tn_temp + fp_temp)
        except ZeroDivisionError:
            fpr_temp = 'NA'
       try:
            ppv_temp = tp_temp / (tp_temp + fp_temp)
        except ZeroDivisionError:
            ppv_temp = 'NA'
       try:
            npv_temp = tn_temp / (tn_temp + fn_temp)
        except ZeroDivisionError:
            npv_temp = 'NA'
       try:
            fdr_temp = fp_temp / (tp_temp + fp_temp)
        except ZeroDivisionError:
            fdr_temp = 'NA'
       try:
            fort_temp = fn_temp / (tn_temp + fn_temp)
        except ZeroDivisionError:
            fort_temp = 'NA'
       try:
            acc_temp = (tp_temp + tn_temp) / (tp_temp + tn_temp + fp_temp + fn_temp)
        except ZeroDivisionError:
            acc_temp = 'NA'
       out_str = str(score) + ', ' +          \
                  str(fp_temp) + ', ' +        \
                  str(tp_temp) + ', ' +        \
                  str(fn_temp) + ', ' +        \
                  str(tn_temp) + ', ' +        \
                  str(fpr_baseline) + ', ' +   \
                  str(tpr_baseline) + ', ' +   \
                  str(fpr_temp) + ', ' +       \
                  str(tpr_temp) + ', ' +       \
                  str(tnr_temp) + ', ' +       \
                  str(fnr_temp) + ', ' +       \
                  str(ppv_temp) + ', ' +       \
                  str(npv_temp) + ', ' +       \
                  str(fdr_temp) + ', ' +       \
                  str(fort_temp) + ', ' +      \
                  str(acc_temp)

        out_handle.write(out_str + '\n')

        out_handle.flush()

        fpr_all.append(fpr_temp)
        tpr_all.append(tpr_temp)

    out_handle.close()

    temp_auc = auc(fpr_all, tpr_all, reorder=False)
    temp_gini = (2 * temp_auc) - 1

   plt.plot(temp_df['CUMSUM_NUM_TXNS_VALID_DIST'],
             temp_df['CUMSUM_NUM_TXNS_FRAUD_DIST'],
             zorder=1, linestyle='-', color='blue', linewidth=3, label='ROC Curve')
    plt.plot(temp_df['MODEL_SCORE_NR_SCALE_INV'].tolist() + [1.1],
             temp_df['MODEL_SCORE_NR_SCALE_INV'].tolist() + [1.1],
             zorder=2, linestyle='-', color='grey', linewidth=3, label='Baseline')
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.title('ROC Curve')
    plt.legend(loc='upper left')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.text(0.5, 0.2, 'AUC: ' + str(auc), horizontalalignment='center',
             fontsize=12, multialignment='left')
    plt.savefig(out_directory + out_model + '_roc_' + temp_period + '.png')
    plt.show()
    plt.close()

    print(str(temp_period) + ' AUC: ' + str(temp_auc) + ' , Gini: ' + str(temp_gini))

    return [temp_auc, temp_gini]


# Configurations
in_dir = '/user/data/'  Note: I modified this input

# in_file = 'Sample_file1.csv'
in_file = 'Sample_file2.csv' Note: I modified this input as well

# model_type = 'credit_combined'
# model_type = 'credit_real_time'
# model_type = 'credit_quasi'
# model_type = 'credit_batch'

model_type = 'debit_pin'
# model_type = 'debit_sign'

out_dir = '/user/results/'  Note: I modified this output

out_file_t1 = 't1.csv'
out_file_t2 = 't2.csv'
out_file_t3 = 't3.csv'
out_file_t4 = 't4.csv'
out_file_t5_pa = 't5_pa.csv'
out_file_t5_pb = 't5_pb.csv'
out_file_t6 = 't6.csv'


# Start Program Timer
start_time = datetime.datetime.now()
start_time_secs = datetime.datetime.strptime(str(start_time),
                                             "%Y-%m-%d %H:%M:%S.%f")
print(str(start_time) + " - Starting analysis...")
sys.stdout.flush()


# Import data
data = pd.read_csv(in_dir + in_file)

data.columns = map(str.upper, data.columns)


data['PERIOD'] = pd.to_datetime(data['PERIOD'], format='%m/%d/%Y')
data['YEAR'] = data['PERIOD'].dt.year
data['MONTH'] = data['PERIOD'].dt.month

cols = data.columns.tolist()
cols = ['PERIOD','YEAR','MONTH'] + [item for item in data.columns.tolist() if item not in ['PERIOD','YEAR','MONTH']]

data = data[cols]
data= data.sort_values(['YEAR', 'MONTH', 'MODEL_SCORE_NR', 'FRAUD_SCORE_TYPE_CD'], ascending=[True, True, True, True])
data = data.reset_index(drop=True)

data['MODEL_SCORE_NR_SCALE'] = data['MODEL_SCORE_NR']/1000
data['MODEL_SCORE_NR_SCALE_INV'] = 0.999 - data['MODEL_SCORE_NR_SCALE']
data[['MODEL_SCORE_NR_BIN', 'MODEL_SCORE_NR_BIN_LABEL']] = data.apply(f_bins, axis=1)  #check this line
data['SCORE_NOT_ZERO'] = data.apply(f_nonzero, axis=1)

data['NUM_TXNS_VALID'] = data[['APPROVED_TRANS_VALID']].sum(axis=1)
data['NUM_TXNS_FRAUD'] = data[['APPROVED_TRANS_FRAUD','FRAUD_DECLINE_TRANS_FRAUD','FRAUD_DECLINE_TRANS_VALID']].sum(axis=1)
data['NUM_TXNS'] = data[['NUM_TXNS_VALID','NUM_TXNS_FRAUD']].sum(axis=1)

data['SALES_VALID'] = data[['APPROVED_AMT_VALID']].sum(axis=1)
data['SALES_FRAUD'] = data[['APPROVED_AMT_FRAUD','FRAUD_DECLINE_AMT_FRAUD','FRAUD_DECLINE_AMT_VALID']].sum(axis=1)
data['SALES'] = data[['SALES_VALID','SALES_FRAUD']].sum(axis=1)

data['DECLINE_TRANS_FRAUD'] = data[['FRAUD_DECLINE_TRANS_FRAUD','FRAUD_DECLINE_TRANS_VALID']].sum(axis=1)
data['DECLINE_AMT_FRAUD'] = data[['FRAUD_DECLINE_AMT_FRAUD','FRAUD_DECLINE_AMT_VALID']].sum(axis=1)

data_unfiltered = data

if model_type == "credit_real_time":
    data = data[data['FRAUD_SCORE_TYPE_CD'] == 'Real Time']

if model_type == "credit_quasi":
    data = data[data['FRAUD_SCORE_TYPE_CD'] == 'Quasi']

if model_type == "credit_batch":
    data = data[data['FRAUD_SCORE_TYPE_CD'] == 'Batch']

if model_type == "debit_pin":
    data = data[data['FRAUD_SCORE_TYPE_CD'] == 'Pin']

if model_type == "debit_sign":
    data = data[data['FRAUD_SCORE_TYPE_CD'] == 'Signature']

data.to_csv(out_dir + model_type + '_' + "data.csv")



pysql = lambda q: pdsql.sqldf(q, globals())

# Create overall dataframe
 
q_f_all = '''
          select   MODEL_SCORE_NR,
                   MODEL_SCORE_NR_SCALE,
                   MODEL_SCORE_NR_SCALE_INV,
                   MODEL_SCORE_NR_BIN,
                  sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID,
                  sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD,
                  sum(NUM_TXNS) as SUM_NUM_TXNS,
                  sum(SALES_VALID) as SUM_SALES_VALID,
                   sum(SALES_FRAUD) as SUM_SALES_FRAUD,
                  sum(SALES) as SUM_SALES,
                  sum(APPROVED_TRANS_VALID) as SUM_APPROVED_TRANS_VALID,
                  sum(FRAUD_DECLINE_TRANS_VALID) as SUM_FRAUD_DECLINE_TRANS_VALID,
                   sum(NONFRAUD_DECLINE_TRANS_VALID) as SUM_NONFRAUD_DECLINE_TRANS_VALID,
                  sum(APPROVED_TRANS_FRAUD) as SUM_APPROVED_TRANS_FRAUD,
                  sum(FRAUD_DECLINE_TRANS_FRAUD) as SUM_FRAUD_DECLINE_TRANS_FRAUD,
                  sum(NONFRAUD_DECLINE_TRANS_FRAUD) as SUM_NONFRAUD_DECLINE_TRANS_FRAUD,
                   
                  sum(APPROVED_AMT_VALID) as SUM_APPROVED_AMT_VALID,
                  sum(FRAUD_DECLINE_AMT_VALID) as SUM_FRAUD_DECLINE_AMT_VALID,
                  sum(NONFRAUD_DECLINE_AMT_VALID) as SUM_NONFRAUD_DECLINE_AMT_VALID,
                  sum(APPROVED_AMT_FRAUD) as SUM_APPROVED_AMT_FRAUD,
                  sum(FRAUD_DECLINE_AMT_FRAUD) as SUM_FRAUD_DECLINE_AMT_FRAUD,
                  sum(NONFRAUD_DECLINE_AMT_FRAUD) as SUM_NONFRAUD_DECLINE_AMT_FRAUD
          from     data
          group by MODEL_SCORE_NR, MODEL_SCORE_NR_SCALE, MODEL_SCORE_NR_SCALE_INV, MODEL_SCORE_NR_BIN
          order by MODEL_SCORE_NR, MODEL_SCORE_NR_SCALE, MODEL_SCORE_NR_SCALE_INV, MODEL_SCORE_NR_BIN
          ;
          '''
 
data_all = pysql(q_f_all)

# Create monthly dataframe

q_f_monthly = '''
              select   YEAR,
                       MONTH,MODEL_SCORE_NR_SCALE,
                       MODEL_SCORE_NR,
                       MODEL_SCORE_NR_SCALE_INV,
                       MODEL_SCORE_NR_BIN,
                       sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID,
                       sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD,
                       sum(NUM_TXNS) as SUM_NUM_TXNS,
                       sum(SALES_VALID) as SUM_SALES_VALID,
                       sum(SALES_FRAUD) as SUM_SALES_FRAUD,
                       sum(SALES) as SUM_SALES,
                       sum(APPROVED_TRANS_VALID) as SUM_APPROVED_TRANS_VALID,
                       sum(FRAUD_DECLINE_TRANS_VALID) as SUM_FRAUD_DECLINE_TRANS_VALID,
                       sum(NONFRAUD_DECLINE_TRANS_VALID) as SUM_NONFRAUD_DECLINE_TRANS_VALID,
                       sum(APPROVED_TRANS_FRAUD) as SUM_APPROVED_TRANS_FRAUD,
                       sum(FRAUD_DECLINE_TRANS_FRAUD) as SUM_FRAUD_DECLINE_TRANS_FRAUD,
                       sum(NONFRAUD_DECLINE_TRANS_FRAUD) as SUM_NONFRAUD_DECLINE_TRANS_FRAUD,
                       sum(APPROVED_AMT_VALID) as SUM_APPROVED_AMT_VALID,
                       sum(FRAUD_DECLINE_AMT_VALID) as SUM_FRAUD_DECLINE_AMT_VALID,
                       sum(NONFRAUD_DECLINE_AMT_VALID) as SUM_NONFRAUD_DECLINE_AMT_VALID,
                       sum(APPROVED_AMT_FRAUD) as SUM_APPROVED_AMT_FRAUD,
                       sum(FRAUD_DECLINE_AMT_FRAUD) as SUM_FRAUD_DECLINE_AMT_FRAUD,
                       sum(NONFRAUD_DECLINE_AMT_FRAUD) as SUM_NONFRAUD_DECLINE_AMT_FRAUD
              from     data
              group by YEAR, MONTH,
                       MODEL_SCORE_NR, MODEL_SCORE_NR_SCALE, MODEL_SCORE_NR_SCALE_INV, MODEL_SCORE_NR_BIN
              order by YEAR, MONTH, PERIOD,
                       MODEL_SCORE_NR, MODEL_SCORE_NR_SCALE, MODEL_SCORE_NR_SCALE_INV, MODEL_SCORE_NR_BIN
              ;
              '''

data_monthly = pysql(q_f_monthly)


# Prep data for KS, AUC, and Gini calculations

data_all = data_prep(data_all, out_directory = out_dir, out_model = model_type, out_str = 'all')
data_monthly = data_monthly.groupby(["YEAR","MONTH"]).apply(data_prep, 
                                                            out_directory = out_dir,
                                                            out_model = model_type,
                                                            out_str = 'monthly')

# Compute KS for whole period and by month

ks_overall = calc_ks(data_all, out_directory = out_dir, out_model = model_type, out_str = 'all')
ks_monthly = data_monthly.groupby(["YEAR","MONTH"]).apply(calc_ks,
                                                          out_directory = out_dir,
                                                          out_model = model_type,
                                                          out_str = 'monthly')

ks_monthly = pd.DataFrame(ks_monthly)
ks_monthly = ks_monthly.reset_index(inplace=False)
ks_monthly.columns = ['YEAR','MONTH','KS']
ks_monthly['YEAR_MONTH'] = ks_monthly['YEAR'].map(str) + '_' + ks_monthly['MONTH'].map(str).str.zfill(2)
ks_monthly['YEAR'] = ks_monthly['YEAR'].astype(int)
ks_monthly['MONTH'] = ks_monthly['MONTH'].astype(int)
ks_monthly['KS'] = ks_monthly['KS'].astype(float)
ks_monthly['YEAR_MONTH'] = ks_monthly['YEAR_MONTH'].astype(str)

# Compute AUC and Gini for whole period and by month

auc_gini_overall = calc_roc(data_all, out_directory = out_dir, out_model = model_type, out_str = 'all')
auc_gini_monthly = data_monthly.groupby(["YEAR","MONTH"]).apply(calc_roc,
                                                                out_directory = out_dir,
                                                                out_model = model_type,
                                                                out_str = 'monthly')

auc_gini_monthly = pd.DataFrame(auc_gini_monthly)
auc_gini_monthly = auc_gini_monthly.reset_index(inplace=False)
auc_gini_monthly.columns = ['YEAR', 'MONTH', 'AUC_GINI']
auc_gini_monthly = pd.concat([auc_gini_monthly, auc_gini_monthly['AUC_GINI'].apply(pd.Series)], axis = 1)
auc_gini_monthly.columns = ['YEAR', 'MONTH', 'AUC_GINI', 'AUC', 'GINI']
auc_gini_monthly['YEAR_MONTH'] = auc_gini_monthly['YEAR'].map(str) + '_' + auc_gini_monthly['MONTH'].map(str).str.zfill(2)
auc_gini_monthly['YEAR'] = auc_gini_monthly['YEAR'].astype(int)
auc_gini_monthly['MONTH'] = auc_gini_monthly['MONTH'].astype(int)
auc_gini_monthly['AUC_GINI'] = auc_gini_monthly['AUC_GINI'].astype(str)
auc_gini_monthly['AUC'] = auc_gini_monthly['AUC'].astype(float)
auc_gini_monthly['GINI'] = auc_gini_monthly['GINI'].astype(float)
auc_gini_monthly['YEAR_MONTH'] = auc_gini_monthly['YEAR_MONTH'].astype(str)

# Merge KS and AUC/Gini dataframes

ks_auc_gini_monthly = pd.merge(ks_monthly[['YEAR_MONTH', 'YEAR', 'MONTH', 'KS']],
                               auc_gini_monthly[['YEAR_MONTH', 'YEAR', 'MONTH', 'AUC', 'GINI']],
                               how = 'inner',
                               on = ['YEAR_MONTH', 'YEAR', 'MONTH'])

# Plot KS, AUC, and Gini by month

months = ks_auc_gini_monthly['YEAR_MONTH'].tolist()
ind = np.arange(len(months))
bar_margin = 0.10
bar_width = 0.35

for i in ['KS', 'AUC', 'GINI']:
    fig, ax = plt.subplots()
    ax.bar(ind, ks_auc_gini_monthly[i], width = bar_width, color = "blue")
    ax.margins(bar_margin, None)
    ax.set_xticks(ind + ((bar_width / 2) + bar_margin))
    ax.set_xticklabels(months)
    plt.title(i + ' by Period')
    plt.xlabel('Month')
    plt.ylabel(i + ' Value')
    plt.savefig(out_dir + model_type + '_' + i.lower() + '_Monthly.png')
    plt.show()
    plt.close()


# PMR - Table 1

q_t1 = '''
       select   MODEL_SCORE_NR_BIN,
               sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID,
               sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD
       from     data 
       group by MODEL_SCORE_NR_BIN
       order by MODEL_SCORE_NR_BIN
       ;
       '''

data_t1 = pysql(q_t1)

data_t1['FRAUD_RATE'] = data_t1['SUM_NUM_TXNS_FRAUD']/(data_t1['SUM_NUM_TXNS_VALID'] + data_t1['SUM_NUM_TXNS_FRAUD'])
data_t1 = data_t1.sort_values('MODEL_SCORE_NR_BIN', ascending=False)
data_t1[['CUMSUM_NUM_TXNS_VALID']] = data_t1[['SUM_NUM_TXNS_VALID']].cumsum(axis=0)
data_t1[['CUMSUM_NUM_TXNS_FRAUD']] = data_t1[['SUM_NUM_TXNS_FRAUD']].cumsum(axis=0)
data_t1 = data_t1.sort_values('MODEL_SCORE_NR_BIN', ascending=True)
data_t1['CUMSUM_FRAUD_RATE'] = data_t1['CUMSUM_NUM_TXNS_FRAUD']/(data_t1['CUMSUM_NUM_TXNS_VALID'] + data_t1['CUMSUM_NUM_TXNS_FRAUD'])

data_t1.to_csv(out_dir + model_type + '_' + out_file_t1)


# PMR - Table 2

q_t2 = '''
       select   MODEL_SCORE_NR_BIN,
               sum(SALES_VALID) as SUM_SALES_VALID,
               sum(SALES_FRAUD) as SUM_SALES_FRAUD
       from     data
       group by MODEL_SCORE_NR_BIN
       order by MODEL_SCORE_NR_BIN
       ;
       '''

data_t2 = pysql(q_t2)

data_t2['FRAUD_DOLLAR_MULTIPLIER'] = data_t2['SUM_SALES_VALID']/data_t2['SUM_SALES_FRAUD']
data_t2 = data_t2.sort_values('MODEL_SCORE_NR_BIN', ascending=False)
data_t2[['CUMSUM_SALES_VALID']] = data_t2[['SUM_SALES_VALID']].cumsum(axis=0)
data_t2[['CUMSUM_SALES_FRAUD']] = data_t2[['SUM_SALES_FRAUD']].cumsum(axis=0)
data_t2 = data_t2.sort_values('MODEL_SCORE_NR_BIN', ascending=True)
data_t2['CUMSUM_FRAUD_DOLLAR_MULTIPLIER'] = data_t2['CUMSUM_SALES_VALID']/data_t2['CUMSUM_SALES_FRAUD']

data_t2.to_csv(out_dir + model_type + '_' + out_file_t2)


# PMR - Table 3

q_t3 = '''
       select   MODEL_SCORE_NR_BIN,
               sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD,
               sum(APPROVED_TRANS_FRAUD) as SUM_NUM_TXNS_FRAUD_APPROVED,
               sum(DECLINE_TRANS_FRAUD) as SUM_NUM_TXNS_FRAUD_DECLINED
       from     data
       group by MODEL_SCORE_NR_BIN
       order by MODEL_SCORE_NR_BIN
       ;
       '''

data_t3 = pysql(q_t3)

data_t3 = data_t3.sort_values('MODEL_SCORE_NR_BIN', ascending=False)
data_t3[['CUMSUM_NUM_TXNS_FRAUD']] = data_t3[['SUM_NUM_TXNS_FRAUD']].cumsum(axis=0)
data_t3[['CUMSUM_NUM_TXNS_FRAUD_APPROVED']] = data_t3[['SUM_NUM_TXNS_FRAUD_APPROVED']].cumsum(axis=0)
data_t3[['CUMSUM_NUM_TXNS_FRAUD_DECLINED']] = data_t3[['SUM_NUM_TXNS_FRAUD_DECLINED']].cumsum(axis=0)

data_t3 = data_t3.sort_values('MODEL_SCORE_NR_BIN', ascending=True)

data_t3.to_csv(out_dir + model_type + '_' + out_file_t3)


# PMR - Table 3 - Addendum

def data_prep_addendum(group, out_directory, out_model, out_str):

    temp_df = group

    if out_str == 'monthly':
       temp_year = list(set(temp_df['YEAR'].tolist()))[0]
       temp_month = list(set(temp_df['MONTH'].tolist()))[0]
       temp_period = str(temp_year) + '_' + str(temp_month).zfill(2)
    else:
        temp_year = 'Overall'
       temp_month = 'Overall'
       temp_period = 'Overall'

   temp_df = temp_df[['MODEL_SCORE_NR_BIN', 'MODEL_SCORE_NR_BIN_LABEL',
                       'NUM_TXNS_FRAUD', 'APPROVED_TRANS_FRAUD', 'DECLINE_TRANS_FRAUD']]

    temp_df = temp_df.sort_values('MODEL_SCORE_NR_BIN', ascending=True)
    temp_df2 = temp_df.groupby(['MODEL_SCORE_NR_BIN', 'MODEL_SCORE_NR_BIN_LABEL'], sort=False).sum()
    temp_df2.reset_index(level=['MODEL_SCORE_NR_BIN', 'MODEL_SCORE_NR_BIN_LABEL'], inplace=True)
    temp_df2.columns = ['MODEL_SCORE_NR_BIN', 'MODEL_SCORE_NR_BIN_LABEL',
                        'SUM_NUM_TXNS_FRAUD', 'SUM_NUM_TXNS_FRAUD_APPROVED', 'SUM_NUM_TXNS_FRAUD_DECLINED']

    temp_df2 = temp_df2.sort_values('MODEL_SCORE_NR_BIN', ascending=False)
    temp_df2[['CUMSUM_NUM_TXNS_FRAUD']] = temp_df2[['SUM_NUM_TXNS_FRAUD']].cumsum(axis=0)
    temp_df2[['CUMSUM_NUM_TXNS_FRAUD_APPROVED']] = temp_df2[['SUM_NUM_TXNS_FRAUD_APPROVED']].cumsum(axis=0)
    temp_df2[['CUMSUM_NUM_TXNS_FRAUD_DECLINED']] = temp_df2[['SUM_NUM_TXNS_FRAUD_DECLINED']].cumsum(axis=0)

    temp_df2 = temp_df2.sort_values('MODEL_SCORE_NR_BIN', ascending=True)
    temp_df2['SUM_APPROVED_DECLINED'] = temp_df2['SUM_NUM_TXNS_FRAUD_APPROVED'] / temp_df2['SUM_NUM_TXNS_FRAUD_DECLINED']
    temp_df2['CUMSUM_APPROVED_DECLINED'] = temp_df2['CUMSUM_NUM_TXNS_FRAUD_APPROVED'] / temp_df2['CUMSUM_NUM_TXNS_FRAUD_DECLINED']

    temp_df2['YEAR_MONTH'] = temp_period 

    temp_df2 = temp_df2[['YEAR_MONTH', 'MODEL_SCORE_NR_BIN', 'MODEL_SCORE_NR_BIN_LABEL',
                         'SUM_APPROVED_DECLINED', 'CUMSUM_APPROVED_DECLINED']]

    return(temp_df2)


data_addendum_all = data_prep_addendum(data, out_directory = out_dir, out_model = model_type, out_str = 'all')

data_addendum_monthly = data.groupby(["YEAR", "MONTH"]).apply(data_prep_addendum,
                                                      out_directory = out_dir,
                                                      out_model = model_type,
                                                      out_str = 'monthly')
data_addendum_monthly.reset_index(level=["YEAR", "MONTH"], inplace=True)

months = data_addendum_monthly['YEAR_MONTH'].tolist()
months_u = []
month_u = [months_u.append(item) for item in months if item not in months_u]

bins = data_addendum_monthly['MODEL_SCORE_NR_BIN_LABEL'].tolist()

ind = np.arange(len(months_u))
bar_margin = 0.05
bar_width = 0.15


# fig = plt.figure(figsize=(10,18))
fig = plt.figure(figsize=(15,15))
fig.suptitle("Fraud Approved-to-Declined Ratio by Score Bin (Scores 700+)", fontsize=22)
fig_dict = OrderedDict()

for jdx, model_val in enumerate(['SUM_APPROVED_DECLINED', 'CUMSUM_APPROVED_DECLINED'], start=1):

    if model_val == "SUM_APPROVED_DECLINED":
        model_val_str = "Sum"
   elif model_val == "CUMSUM_APPROVED_DECLINED":
        model_val_str = "Cumsum"
   else:
        model_val_str = "ERROR"

   fig = plt.figure(figsize=(15,15))
    fig.suptitle("Fraud Approved-to-Declined Ratio by Score Bin (Scores 700+) - " + model_val_str, fontsize=22)
    fig_dict = OrderedDict()

    for idx, bin_val in enumerate([15, 16, 17, 18, 19, 20], start=1):

        fig_dict['ax' + str(idx)] = fig.add_subplot(2, 3, idx)
        _data = data_addendum_monthly[data_addendum_monthly['MODEL_SCORE_NR_BIN_LABEL'] == bins[bin_val]][model_val]
        fig_dict['ax' + str(idx)].set_title(bins[bin_val])
        fig_dict['ax' + str(idx)].bar(ind,_data, width = bar_width, color = "blue")
        fig_dict['ax' + str(idx)].margins(bar_margin, None)
        fig_dict['ax' + str(idx)].set_ylim([0.0,6.0])
        fig_dict['ax' + str(idx)].set_xticks(ind + ((bar_width / 2) + bar_margin))
        fig_dict['ax' + str(idx)].set_xticklabels(months_u, rotation = 90)

    plt.savefig(out_dir + model_type + '_ratios_Monthly_' + model_val_str + '.png')
    plt.show()
    plt.close()


# PMR - Table 4

q_t4 = '''
       select   MODEL_SCORE_NR_BIN,
               sum(SALES_FRAUD) as SUM_SALES_FRAUD,
               sum(APPROVED_AMT_FRAUD) as SUM_SALES_FRAUD_APPROVED,
               sum(DECLINE_AMT_FRAUD) as SUM_SALES_FRAUD_DECLINED
       from     data
       group by MODEL_SCORE_NR_BIN
       order by MODEL_SCORE_NR_BIN
       ;
       '''

data_t4 = pysql(q_t4)

data_t4 = data_t4.sort_values('MODEL_SCORE_NR_BIN', ascending=False)
data_t4[['CUMSUM_SALES_FRAUD']] = data_t4[['SUM_SALES_FRAUD']].cumsum(axis=0)
data_t4[['CUMSUM_SALES_FRAUD_APPROVED']] = data_t4[['SUM_SALES_FRAUD_APPROVED']].cumsum(axis=0)
data_t4[['CUMSUM_SALES_FRAUD_DECLINED']] = data_t4[['SUM_SALES_FRAUD_DECLINED']].cumsum(axis=0)

data_t4 = data_t4.sort_values('MODEL_SCORE_NR_BIN', ascending=True)

data_t4.to_csv(out_dir + model_type + '_' + out_file_t4)


# PMR - Table 5 (Panel A)

q_t5_pa = '''
          select   FRAUD_SCORE_TYPE_CD,
                  sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID,
                   sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD,
                  sum(APPROVED_TRANS_FRAUD) as SUM_NUM_TXNS_FRAUD_APPROVED,
                  sum(DECLINE_TRANS_FRAUD) as SUM_NUM_TXNS_FRAUD_DECLINED
          from     data_unfiltered
          group by FRAUD_SCORE_TYPE_CD
          order by FRAUD_SCORE_TYPE_CD
          ;
          '''

data_t5_pa = pysql(q_t5_pa)

data_t5_pa.to_csv(out_dir + model_type + '_' + out_file_t5_pa)


# PMR - Table 5 (Panel B)

q_t5_pb = '''
          select   SCORE_NOT_ZERO,
                   FRAUD_SCORE_TYPE_CD,
                  sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID,
                  sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD,
                  sum(APPROVED_TRANS_FRAUD) as SUM_NUM_TXNS_FRAUD_APPROVE,
                  sum(DECLINE_TRANS_FRAUD) as SUM_NUM_TXNS_FRAUD_DECLINED
          from     data_unfiltered
          group by SCORE_NOT_ZERO,FRAUD_SCORE_TYPE_CD
          order by SCORE_NOT_ZERO,FRAUD_SCORE_TYPE_CD
          ;
          '''

data_t5_pb = pysql(q_t5_pb)

data_t5_pb.to_csv(out_dir + model_type + '_' + out_file_t5_pb)


# PMR - Table 6

q_t6_base = '''
            select   MODEL_SCORE_NR_BIN,
                    sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID_BASE,
                    sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD_BASE
            from     data
            where    YEAR = '2017'
            and      MONTH = '2'
            group by MODEL_SCORE_NR_BIN
            order by MODEL_SCORE_NR_BIN
            ;
            '''

data_t6_base = pysql(q_t6_base)


q_t6_latest = '''
              select   MODEL_SCORE_NR_BIN,
                      sum(NUM_TXNS_VALID) as SUM_NUM_TXNS_VALID_LATEST,
                      sum(NUM_TXNS_FRAUD) as SUM_NUM_TXNS_FRAUD_LATEST
              from     data
              where    YEAR = '2017'
              and      MONTH = '5'
              group by MODEL_SCORE_NR_BIN
              order by MODEL_SCORE_NR_BIN
              ;

              '''

data_t6_latest = pysql(q_t6_latest)


data_t6 = pd.merge(data_t6_latest, data_t6_base,
                   left_on='MODEL_SCORE_NR_BIN', right_on='MODEL_SCORE_NR_BIN', how='inner')

print (data_t6)

data_t6['SUM_NUM_TXNS_LATEST'] = data_t6[['SUM_NUM_TXNS_VALID_LATEST','SUM_NUM_TXNS_VALID_LATEST']].sum(axis=1)
data_t6['SUM_NUM_TXNS_BASE'] = data_t6[['SUM_NUM_TXNS_VALID_BASE','SUM_NUM_TXNS_VALID_BASE']].sum(axis=1)

total_latest, total_base = data_t6[['SUM_NUM_TXNS_LATEST','SUM_NUM_TXNS_BASE']].sum(axis=0)

data_t6['SCORING_PCT'] = data_t6['SUM_NUM_TXNS_LATEST']/total_latest
data_t6['TRAINING_PCT'] = data_t6['SUM_NUM_TXNS_BASE']/total_base

data_t6['DIFF'] = data_t6['SCORING_PCT'] - data_t6['TRAINING_PCT']
data_t6['LOG'] = np.log(data_t6['SCORING_PCT']/data_t6['TRAINING_PCT'])

data_t6['PSI'] = data_t6['DIFF'] * data_t6['LOG']

data_t6.to_csv(out_dir + model_type + '_' + out_file_t6)

psi = data_t6['PSI'].sum(axis=0)

print('PSI: ' + str(psi))
