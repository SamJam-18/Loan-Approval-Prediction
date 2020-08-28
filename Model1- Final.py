import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pprint


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
def train_test(X, y, test_size = .20, sample = 'over'):
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


#print (train_test(X, y))

#creating the model chosen(RandomForest-over sampling)
def RF(X,y, sample = 'under'):

  X_train, X_test, y_train, y_test = train_test(X,y, sample = sample)

  K = RandomForestClassifier()

  #here is where we tune hyperparameters

  grid_= { 'max_features' : ['auto'],
          'max_depth': [3],
          'min_samples_split' : [4, 3, 2],
          'min_samples_leaf': [2, 3, 4],
          'n_estimators' : [80]
          }

  grid = GridSearchCV(K, grid_)
  grid.fit(X_train, y_train)

  print (grid.best_params_)

  y_preds = grid.predict(X_test)


  acc = accuracy_score(y_test, y_preds)
  prec = precision_score(y_test, y_preds)
  rec = recall_score(y_test, y_preds)
  f1 = f1_score(y_test, y_preds)



  print('Accuracy: %.3f' % acc)
  print('Precision: %.3f' % prec)
  print('Recall: %.3f' % rec)
  print('f1: %.3f' % f1)

  return (grid.fit(X_train, y_train))


#(RF(X,y))

def Fin_mod_1(X, y):

  #X = X.drop(['Delinquences last 2 yrs', 'Loan Length', 'Home: Rent', 'Home: Mortage', 'Home: Own', 'Home: Any', 'Home: None', 'Debt to Income ratio', 'Employment Length' ], axis =1)

  X_train, X_test, y_train, y_test = train_test(X,y, sample = 'over')

  RF = RandomForestClassifier()

  #here is where we tune hyperparameters

  grid_= { 'max_features' : ['auto'],
          'max_depth': [3],
          'min_samples_split' : [4],
          'min_samples_leaf': [4],
          'n_estimators' : [80]
          }

  grid = GridSearchCV(RF, grid_)
  grid.fit(X_train, y_train)



  y_preds = grid.predict_proba(X_test)

  #return (grid.fit(X_train, y_train))


  acc = accuracy_score(y_test, y_preds)
  prec = precision_score(y_test, y_preds)
  rec = recall_score(y_test, y_preds)
  f1 = f1_score(y_test, y_preds)


  print('Accuracy: %.3f' % acc)
  print('Precision: %.3f' % prec)
  print('Recall: %.3f' % rec)
  print('f1: %.3f' % f1)



Fin_mod_1(X,y)


#calc fearue importance (The impurity-based feature importances)
def feat(X,y):

  K = Fin_mod_1(X,y)

  importances = K.best_estimator_.feature_importances_
  sorted_idx = importances.argsort()[::-1]
  features = X.columns

  X_axis = []
  Y_axis = []
  for x in sorted_idx:
    X_axis.append(importances[x])
    Y_axis.append(features[x])





  sns.axes_style('whitegrid', {'ytick.left': True, 'axes.spines.right': False, 'axes.spines.top' : False})
  sns.set(style="whitegrid",font="sans-serif")
  sns.set(font_scale=1.2)
  sns.barplot(X_axis, Y_axis, palette="GnBu_d")


  plt.title('Feature Importances - Impurity Based')
  plt.xlabel('Relative Importance')
  plt.savefig('Model_plots/Feature Importances RF.png', bbox_inches='tight')

  #plt.show()

#feat(X,y)

#mean decrease impurity plotting to get feature importance
def MDI(X,y, sample = 'over'):

  K = Fin_mod_1(X,y)

  X_dummies = pd.get_dummies(X)

  feat_scores = pd.DataFrame({'Fraction of Samples Affected' : K.best_estimator_.feature_importances_},index = X_dummies.columns)
  feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected', ascending=False)

  x_axis = feat_scores['Fraction of Samples Affected'].tolist()
  y_axis = feat_scores.index.tolist()

  sns.set(style="whitegrid",font="sans-serif")
  sns.set(font_scale=1.2)
  sns.barplot(x_axis, y_axis, palette='GnBu_d', saturation = 1, )

  plt.title('Feature Importances - MDI')
  plt.savefig('Model_plots/Feature Importances_MDI_RF.png', bbox_inches='tight')
  plt.show()

  return(y_axis)

MDI(X,y, sample = 'over')


#Final Model with all the changes to the data and tuned


#benifit matrix
def profit(df, X, y):

  profits = []

  gain = [454, 880, 1154, 1406, 1943,
            2304, 3143, 4091, 5961]

  loss = [-1125, -2110, -2756, -3502, -4373, -5149, -6220, -7532, -9509]

  range_loan = ['500 - 3200', '3200 - 5000', '5000 - 6000', '6000-8000', '8000-10,000', '10,000-11,625', '11,625-14,000', '14,000-16,100', '16,100-21,250']

  df = df.sort_values(by=['Loan Amount'])
  df = np.array_split(df, 10)


  for x in range(len(loss)): #run through each range of loan price

    cost_benefit_matrix = np.array([[gain[x], 0],
                                      [loss[x], 0]])

    data = df[x]
    y = data['status']
    X = data.drop('status', axis = 1)

    new_pred, y_test = Fin_mod_1(X,y)

  #create a confusion matrix
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, new_pred)
    conf_matrix = np.array([[tp, fp], [fn, tn]])


  #create a list of profits
    profit = np.sum(conf_matrix * cost_benefit_matrix)

    print (range_loan[x])
    print (profit)
    print (len(new_pred))

  new_pred, y_test = Fin_mod_1(X,y)

  cost_benefit_matrix = np.array([[3109, 0],
                                      [-5583, 0]])

  [[tn, fp], [fn, tp]] = confusion_matrix(y_test, new_pred)
  conf_matrix = np.array([[tp, fp], [fn, tn]])

  profit = np.sum(conf_matrix * cost_benefit_matrix)

  print ('TOTAL')
  print (profit)

#profit(df, X, y)