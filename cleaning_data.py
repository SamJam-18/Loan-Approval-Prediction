import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

sns.set(style="ticks")
raw_data = pd.read_csv('Loan.csv')


# droped all the rows that have In Grace period in the Status Column
raw_data = raw_data[raw_data['Status'] != 'In Grace Period']

'''
 from initial EDA found that:
  Droped Columns:
      - Months Since Last Delinquency, total = 612349 and Nan = 30922
      about 19 % and cannot infere that the Nan values = zero so I droped
      the column

      -'Revolving Line Utilization' does not have any impact as it shows the
      current limits an applicant is using (after being approved)

      - Months Since Last Record - had 44379 NuNs so I dropped it
  Droped Rows:
      - found 17 applicants did not have a fico score so I dropped applicants

      -found that 29 apllicants had Nan in all columns shown below:
      ['Open CREDIT Lines', 'Total CREDIT Lines', 'Revolving Line Utilization',
       'Inquiries in the Last 6 Months', 'Accounts Now Delinquent',
       'Delinquencies (Last 2 yrs)', 'Months Since Last Delinquency',
       'Public Records On File'] - dropped applicants
'''


data = raw_data.drop(['Months Since Last Delinquency', 'Revolving Line Utilization', 'Months Since Last Record'], axis = 1)


data = data[data['Approx. Fico Score'].notna()]
data = data[data['Open CREDIT Lines'].notna()]

# change Object types using dummy
## Home ownership
data = pd.get_dummies(data, columns = ['Home Ownership'])


#Change Status to a True or False Boolean
data['Status'] = data['Status'].map({'Not Delinquent': 0, 'Delinquent': 1})

#change Loan Lenths months to an interger
data['Loan Length'] = data['Loan Length'].str.replace('months', '').astype(int)

#grouby the delinquent and not delinquent
data_stat = data.groupby('Status').describe().reset_index()
#save a csv file with the infromation
'''
dt = data.drop(['Status', 'Home Ownership_ANY',
       'Home Ownership_MORTGAGE', 'Home Ownership_NONE', 'Home Ownership_OWN',
       'Home Ownership_RENT'], axis = 1)
columns = dt.columns

for x in columns:
    data_stat[x].to_csv('describe_data.csv',mode='a',line_terminator = '\n' + x)

print(data.columns)
'''

#change column names to be easier to write and read

data.columns = ['total_amount', 'loan_lenght', 'month_pay',
       'debt_income_ratio', 'month_income', 'fico_score',
       'open_credit_lines', 'tot_credit_lines', 'rev_credit_bal',
       'inquiries_6_month', 'acc_now_delinquent',
       'del_last_2yrs', 'public_records',
       'employ_length', 'status', 'home_any',
       'home_mortage', 'home_none', 'home_own',
       'home_rent']

print (data.columns)

data.to_csv('clean_sdata.csv',index = False, header=True)

