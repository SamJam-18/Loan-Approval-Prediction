import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, confusion_matrix, plot_confusion_matrix
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

# Running the model to tune the hyperparameters
def run_model2(X, y, sample):
  X_train, X_test, y_train, y_test = train_test(X,y, sample = sample)

  X = X.drop(['Delinquences last 2 yrs', 'Loan Length', 'Home: Rent', 'Home: Mortage', 'Public Records on File', 'Home: Own', 'Home: Any', 'Home: None'], axis =1)

  K = RandomForestClassifier()

  #here is where we tune hyperparameters

          #'max_leaf_nodes': None,
          #'min_impurity_decrease': 0.0,
          #'min_impurity_split': None,
          #'min_samples_split': 2,
          #'min_weight_fraction_leaf': 0.0,
          #'n_jobs': 1,
          #'oob_score': False,
          #'random_state': 1,
          #'verbose': 0,
          #'warm_start': False}
          #'class_weight': [None],
          #'criterion': ['gini'],

  grid_= {}

  grid = GridSearchCV(K, grid_)
  grid.fit(X_train, y_train)

  print (grid.best_params_)

  y_pred = grid.predict_proba(X_test)[:,1] #get the probability of getting 1
  #given a threshold - made a new prediction array to compare with the y_test data

  new_pred = []

  for x in y_pred:
    if x  > .95:
      new_pred.append(1)
    else:
      new_pred.append(0)

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

  return (grid.fit(X_train, y_train), y_pred)

#run_model2(X,y, sample = 'under')



#calc fearue importance (The impurity-based feature importances)
def feat(X,y):

  K, y_pred = run_model2(X,y, sample = 'over')

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
  plt.savefig('Model_plots/Feature Importances RFP.png', bbox_inches='tight')

  plt.show()

#feat(X,y)

#mean decrease impurity plotting to get feature importance
def MDI(X,y, sample = 'over'):

  K, y_pred = run_model2(X,y, sample = 'over')

  X_dummies = pd.get_dummies(X)

  feat_scores = pd.DataFrame({'Fraction of Samples Affected' : K.best_estimator_.feature_importances_},index=X_dummies.columns)
  feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected', ascending=False)

  x_axis = feat_scores['Fraction of Samples Affected'].tolist()
  y_axis = feat_scores.index.tolist()

  sns.set(style="whitegrid",font="sans-serif")
  sns.set(font_scale=1.2)
  sns.barplot(x_axis, y_axis, palette='GnBu_d', saturation = 1, )

  plt.title('Feature Importances - MDI')
  plt.savefig('Model_plots/Feature Importances_MDI_RFP.png', bbox_inches='tight')
  #plt.show()

  return(y_axis)

#(MDI(X,y, sample = 'over'))

#final model with hyperparameters
def Fin_mod_2(X, y):

  #X = X.drop(['Delinquences last 2 yrs', 'Loan Length', 'Home: Rent', 'Home: Mortage', 'Public Records on File', 'Home: Own', 'Home: Any', 'Home: None'], axis =1)

  X_train, X_test, y_train, y_test = train_test(X,y, sample = 'under')

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

  y_pred = grid.predict_proba(X_test)[:,1] #get the probability of getting 1 aka delinquent
  #given a threshold - made a new prediction array to compare with the y_test data

  new_pred = []
  for x in y_pred:
    if x  > .70:
      new_pred.append(1)
    else:
      new_pred.append(0)

  #return (new_pred, y_test)

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

  return (new_pred, y_test)

#Fin_mod_2(X,y)



'''
#calculate the best threshold - just went with .95 because I wanted a high percision score
def thres(df,X,y):
  #get a range of thresholds
  thresholds = np.linspace(0, 1, 100).tolist()



  gain = [454, 880, 1154, 1406, 1943,
            2304, 3143, 4091, 5961]

  gain_n = [-454, -880, -1154, -1406, -1943,
            -2304, -3143, -4091, -5961]

  loss = [-1125, -2110, -2756, -3502, -4373, -5149, -6220, -7532, -9509]

  range_loan = ['500 - 3200', '3200 - 5000', '5000 - 6000', '6000-8000', '8000-10,000', '10,000-11,625', '11,625-14,000', '14,000-16,100', '16,100-21,250']
  df = df.sort_values(by=['Loan Amount'])
  df = np.array_split(df, 10)

  sns.axes_style('whitegrid', {'ytick.left': True, 'axes.spines.right': False, 'axes.spines.top' : False})
  sns.set(style="whitegrid",font="sans-serif")
  #sns.set(font_scale=1.2)
  sns.set_palette('GnBu_d')


  for z in range(len(gain)): #run through each range of loan price
    cost_benefit_matrix = np.array([[0, gain_n[z]],
                                      [loss[z], gain[z]]])

    data = df[z]
    y = data['status']
    X = data.drop('status', axis = 1)

    y_pred, y_test = Fin_mod_2(X,y)
    y_pred = y_pred[:, 0]
    profits = []
    for thresh in thresholds: #run though each threshold
      new_pred = []
      for x in y_pred: #run through each y_pred
        if x  > thresh:
          new_pred.append(1)
        else:
          new_pred.append(0)
      #create a confusion matrix
      [[tn, fp], [fn, tp]] = confusion_matrix(y_test, new_pred)
      conf_matrix = np.array([[tp, fp], [fn, tn]])

      #create a list of profits
      profit = int(np.sum(conf_matrix * cost_benefit_matrix))

      profits.append(profit)

    label = range_loan[z]

    sns.lineplot(thresholds, profits, label = label)
    #plt.vlines(max_thresh, min_, max_ + 10000, label='Max. Profit Thresh. : %.2f' % max_thresh, colors= 'Bu_d', )

  plt.ylabel('Profit $')
  plt.xlabel('Probability Threshold')
  plt.title('Profit Curve')
  plt.legend(loc=4)

  plt.savefig('Model_plots/ROC Curve.png', bbox_inches='tight')

  plt.show()

#thres(df,X,y)
'''

def profit(df, X, y):

  profits = []

  gain_n = [-454, -880, -1154, -1406, -1943,
            -2304, -3143, -4091, -5961]

  gain = [454, 880, 1154, 1406, 1943,
            2304, 3143, 4091, 5961]

  loss = [-1125, -2110, -2756, -3502, -4373, -5149, -6220, -7532, -9509]

  range_loan = ['500 - 3200', '3200 - 5000', '5000 - 6000', '6000-8000', '8000-10,000', '10,000-11,625', '11,625-14,000', '14,000-16,100', '16,100-21,250']

  df = df.sort_values(by=['Loan Amount'])
  df = np.array_split(df, 10)


  for x in range(len(loss)): #run through each range of loan price

    cost_benefit_matrix = np.array([[0, gain_n[x]],
                                      [loss[x], gain[x]]])

    data = df[x]
    y = data['status']
    X = data.drop('status', axis = 1)

    new_pred, y_test = Fin_mod_2(X,y)

  #create a confusion matrix
    [[tn, fp], [fn, tp]] = confusion_matrix(y_test, new_pred)
    conf_matrix = np.array([[tp, fp], [fn, tn]])


  #create a list of profits
    profit = np.sum(conf_matrix * cost_benefit_matrix)

    print (range_loan[x])
    print ('-----')
    print (profit)
    print ('-----')

profit(df, X, y)


new_pred, y_test = Fin_mod_2(X,y)

cost_benefit_matrix = np.array([[0, -3109],
                              [-5583, 3109]])

[[tn, fp], [fn, tp]] = confusion_matrix(y_test, new_pred)
conf_matrix = np.array([[tp, fp], [fn, tn]])

profit = np.sum(conf_matrix * cost_benefit_matrix)

print ('TOTAL')
print (profit)


sns.heatmap(conf_matrix, annot=True, cmap = 'Blues')
plt.show()
plt.savefig('Model_plots/2_conf.png', bbox_inches='tight')