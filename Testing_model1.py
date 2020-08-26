import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

df = pd.read_csv('clean_sdata.csv')

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
def run_model (X, y , models, names, sample):
  X_train, X_test, y_train, y_test = train_test(X,y, sample = sample)

  for x in range(len(models)):
    R = models[x]


    R.fit(X_train, y_train)
    y_preds = R.predict(X_test)

    acc = accuracy_score(y_test, y_preds)
    prec = precision_score(y_test, y_preds)
    rec = recall_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)

    print (f'Model: {names[x]}')
    print ('------------------')
    print('Accuracy: %.3f' % acc)
    print('Precision: %.3f' % prec)
    print('Recall: %.3f' % rec)
    print('f1: %.3f' % f1)



# below is what I ran to find the best model
'''
m = [KNeighborsClassifier(n_neighbors=3), RandomForestClassifier(), DecisionTreeClassifier(random_state=0)]
n = ['KNeighborsClassifier', 'RandomForestClassifier', 'DecisionTreeClassifier']

run_model(X, y, m, n, 'under')
run_model(X, y, m, n, 'over')
'''

#creating the model chosen(RandomForest-overfitting)
def RF(X,y, sample = 'over'):

  X_train, X_test, y_train, y_test = train_test(X,y, sample = sample)

  K = RandomForestClassifier()
  return (K.fit(X_train, y_train))

#mean decrease impurity plotting
def MDI(X,y, sample = 'over'):

  K = RF(X,y, sample = 'over')

  X_dummies = pd.get_dummies(X)

  feat_scores = pd.DataFrame({'Fraction of Samples Affected' : K.feature_importances_},index=X_dummies.columns)
  feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected', ascending=False)

  x_axis = feat_scores['Fraction of Samples Affected'].tolist()
  #wanted to change the labels so I manualley imputed them based off there order :(
  y_axis = ['Rev. Credit Balance', 'Monthly Payment', 'Debt to Income Ratio',
            'Monthly Income', 'Total Credit Lines', 'Loan Amount', 'Inquiries last 6 months',
            'Open Credit Line', 'Employment Length', 'Fico Score', 'Loan Length',
            'Delinquencies last 2 yrs', 'Home: Rents', 'Home: Mortage', 'Public Records on File',
            'Home: Own', 'Home: Any', 'Home: None' ]

  sns.barplot(x_axis, y_axis)
  sns.set(rc={'figure.figsize':(20,15)})
  plt.title('Feature Importances - MDI')
  plt.savefig('Model_plots/Feature Importances_MDI_RF.png', bbox_inches='tight')

# MDI(X,y, sample = 'over')

#mean decreae accuracy plotting
def MDA(X,y, sample = 'over'):

  scores = defaultdict(list)
  X_dummies = pd.get_dummies(X)
  splitter = ShuffleSplit(100, test_size=.3)
  rf = RandomForestClassifier()
  for train_idx, test_idx in splitter.split(X_dummies, y):

      X_train, X_test = X_dummies.iloc[train_idx], X_dummies.iloc[test_idx]
      y_train, y_test = y[train_idx], y[test_idx]

      rf.fit(X_train, y_train)
      acc = accuracy_score(y_test, rf.predict(X_test))

      for i, name in enumerate():
          X_train_copy = X_train.copy()
          X_train_copy = X_train_copy.drop(names[name], axis=1)

          X_test_copy = X_test.copy()
          X_test_copy = X_test_copy.drop(names[name], axis=1)

          rf.fit(X_train_copy, y_train)

          shuff_acc = accuracy_score(y_test, rf.predict(X_test_copy))
          scores[names_keys[i]].append((acc-shuff_acc)/acc)

  score_series = pd.DataFrame(scores).mean()

  scores = pd.DataFrame({'Mean Decrease Accuracy' : score_series})

  scores.sort_values(by='Mean Decrease Accuracy').plot(kind='barh')
  plt.title('RF Feature Importances - MDA')
  plt.savefig('Model_plots/FeatureImportances_MDA_RF.png')

MDA(X,y)

'''
# feature importance graph

X_dummies = pd.get_dummies(X)
feat_scores = pd.DataFrame({'Fraction of Samples Affected' : K.feature_importances_}, index=X_dummies.columns)
feat_scores = feat_scores.sort_values(by='Fraction of Samples Affected')
feat_scores.plot(kind='barh')
plt.title('Feature Importances - MDI')
plt.savefig('EDA-graphs/Feature Importances_MDI_.png')


A, B, C, D =train_test(X, y)

'''
