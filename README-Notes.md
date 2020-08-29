# Loan-Approval-Prediction
Create a  model to predict if an applicant will be come delinquent on a loan payment, given the information provided in a typical loan application.

## Notes

### Model predicting wether or not applicant can pay back loan

Model: KNeighborsClassifier - under
------------------
Accuracy: 0.831
Precision: 0.098
Recall: 0.172
f1: 0.125

Model: RandomForestClassifier - under
------------------
Accuracy: 0.928
Precision: 0.316
Recall: 0.018
f1: 0.034

Model: DecisionTreeClassifier - under
------------------
Accuracy: 0.876
Precision: 0.119
Recall: 0.120
f1: 0.119

Model: KNeighborsClassifier - over
------------------
Accuracy: 0.534
Precision: 0.080
Recall: 0.568
f1: 0.141

Model: RandomForestClassifier - over
------------------
Accuracy: 0.642
Precision: 0.108
Recall: 0.596
f1: 0.183

Model: DecisionTreeClassifier - over
------------------
Accuracy: 0.557
Precision: 0.081
Recall: 0.543
f1: 0.142

#### Random Forest Classifier (over sampling) was found to be the best based off of the Recall score - plot showing this
  + {'max_depth': 3, 'max_features': 'auto', 'min_samples_leaf': 4, 'min_samples_split': 4, 'n_estimators': 80}
  Accuracy: 0.608
  Precision: 0.102
  Recall: 0.642
  f1: 0.176

range_loan = ['500 - 3200', '3200 - 5000', '5000 - 6000', '6000-8000', '8000-10,000', '10,000-11,625', '11,625-14,000', '14,000-16,100', '16,100-21,250']


profit = [-355500, -666417, -870210, -1105603, -1380496, -1625369, -1963462, -2377711, -3002100]

### Model Predicting the percentage that the applicant can pay back the loan

in order of thresholds [.9, .8, .95]

Model: RandomForestClassifier - under
------------------
Accuracy: 0.624, 0.851, 0.368
Precision: 0.101, 0.174, 0.077
Recall: 0.589, 0.277, 0.798
f1: 0.173, 0.214, 0.140


Model: DecisionTreeClassifier - under
------------------
Accuracy: 0.878, 0.877, 0.879
Precision: 0.104, 0.118, 0.109
Recall: 0.109, 0.105, 0.120
f1: 0.107, 0.111, 0.114


Model: RandomForestClassifier - over
------------------
Accuracy: 0.076, 0.136, 0.073
Precision: 0.067, 0.075, 0.072
Recall: 0.997, 0.984, 1.000
f1: 0.126, 0.139, 0.134


Model: DecisionTreeClassifier - over
------------------
Accuracy: 0.558, 0.551, 0.552
Precision: 0.083, 0.087, 0.089
Recall: 0.556, 0.561, 0.568
f1: 0.144, 0.150, 0.154

#### Random Forest Classifier (under sampling) was found to be the best based off of the Log Loss score



### Calculated feature importance is same for both models



### Cost Benifit Matrix
- mean loan gross income: 3109.562952875248
- average loss if paid off half off: 5583
