import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

df = pd.read_csv('clean_sdata.csv')


# changed the names of columns for later plotting
df = df.rename(columns={'total_amount' : 'Loan Amount', 'loan_lenght' : 'Loan Length', 'month_pay' : 'Monthly Payment',
        'debt_income_ratio' : 'Debt to Income Ratio','month_income' : 'Monthly Income', 'fico_score' : 'Fico Score',
        'open_credit_lines' : 'Open Credit Lines', 'tot_credit_lines' : 'Total Credit Lines', 'inquiries_6_month' : 'Inquiries 6 Months',
        'del_last_2yrs' : 'Delinquences last 2 yrs', 'public_records' : 'Public Records on File', 'employ_length' : 'Employment Length',
       'status': 'status', 'home_any' : 'Home: Any', 'home_mortage' : 'Home: Mortage', 'home_none' : 'Home: None', 'home_own' : 'Home: Own',
       'home_rent': 'Home: Rent'})

#create my label and data
y = df['status']
X = df.drop('status', axis = 1)

#Create a train_test_split with both over and under sampling
def train_test(X, y, test_size = .20, sample = 'under'):
  #do a train test split
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size)

  #create a train dataframe
  train_data = pd.concat([X_train, y_train], axis=1)

  #seperate into delinquent and non delinquent
  delin = train_data[train_data.status == 1]
  non_delin = train_data[train_data.status == 0]

  #either perform a under or over smapling
  if sample == 'under':
    under_sample = resample(delin,n_samples=len(non_delin)) #gives only delin
    new_sample = pd.concat([under_sample, non_delin]) #merge both delin and non delin together

  elif sample == 'over':
    over_sample = resample(non_delin,n_samples=len(delin)) #gives only non delin
    new_sample = pd.concat([over_sample, delin]) #merge both non delin and delin together


  X_train = new_sample.drop('status', axis = 1)
  y_train = new_sample['status']

  return (X_train, X_test, y_train, y_test)




#create a function that will run through the diffrent models and
## give infomation about the prediction and plot the accuracy/percison/recall
def run_model(X, y , models, names, sample):
  X_train, X_test, y_train, y_test = train_test(X,y, sample = sample)


  for x in range(len(models)):
    R = models[x]
    param_grid = {}
    grid = GridSearchCV(R, param_grid = param_grid)


    grid.fit(X_train, y_train)

    y_preds = grid.predict(X_test)


    acc = accuracy_score(y_test, y_preds)
    prec = precision_score(y_test, y_preds)
    rec = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
'''
    print (f'Model: {names[x]}')
    print ('------------------')
    print('Accuracy: %.3f' % acc)
    print('Precision: %.3f' % prec)
    print('Recall: %.3f' % rec)
    print('f1: %.3f' % f1)
'''


# below is what I ran to find the best model

m = [KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(), DecisionTreeClassifier()]
n = ['KNeighborsClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']

#run_model(X, y, m, n, 'under')
#run_model(X, y, m, n, 'over')

#run_model(X, y, [LogisticRegression(max_iter = 0, solver='lbfgs')], ['RF'], 'over')



#now test the 2nd model
def run_model2(X, y , models, names, sample):
  X_train, X_test, y_train, y_test = train_test(X,y, sample = sample)


  for x in range(len(models)):
    R = models[x]
    param_grid = {}
    grid = GridSearchCV(R, param_grid = param_grid)

    grid.fit(X_train, y_train)

    y_pred = grid.predict_proba(X_test)[:,0] #get the probability of getting 1
    #given a threshold - made a new prediction array to compare with the y_test data

    new_pred = []

    for x in y_pred:
      if x  > .95:
        new_pred.append(0)
      else:
        new_pred.append(1)

    acc = accuracy_score(y_test, new_pred)
    prec = precision_score(y_test, new_pred)
    rec = recall_score(y_test, new_pred)
    f1 = f1_score(y_test, new_pred)

    #print (f'Model: {names[x]}')
    print ('------------------')
    print('Accuracy: %.3f' % acc)
    print('Precision: %.3f' % prec)
    print('Recall: %.3f' % rec)
    print('f1: %.3f' % f1)






# below is what I ran to find the best model - tried to use Loggistic Regression but was getting a lot of errors

x = [ RandomForestClassifier(), DecisionTreeClassifier()]
z = ['RandomForestClassifier', 'DecisionTreeClassifier']

#run_model2(X, y, x, z, 'under')

#run_model2(X, y, x, z, 'over')

#ran this multiple times with diffrent thresholds (.7, .8, .9) to get the best model
