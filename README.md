# Loan-Approval-Prediction
Create a  model to predict if an applicant will be come delinquent on a loan payment, given the information provided in a typical loan application.

## Data
The data used was from already approved loan applicants and because of this was highly imbalanced. It contined a total of 48,216 rows with 19 features, 3,292 where Delinquent while 44,924 where Non Delinquent, this is about a 7 to 100 ratio. Most of the features contained numberals while a few contained strings. 

## Process
Below is the process I used to create and test my model. 
![picture](Model_plots/Process.png)
  
## EDA
While performing EDA I did not find any noticable trends in the scatter matrix. Below you can see there are no significant diffrence between delinquent and non delinquent. This was to be expected as the data was from already approved applicants. 

![picture](EDA_graphs/box1.png)
![picture](EDA_graphs/bow2.png)
![picture](EDA_graphs/bax3.png)
![picture](EDA_graphs/box4.png)

## Cleaning
While looking through the data I took out Revolving Credit Balance, Revolving Line Utilization, Account Now delinquent, Months since last Delinquency, Months since last record as they were either laking in data or did not pertain to a loan application, I also removed any rows that had ant Nan values. After this I was left with a total of 48,170, 44,889 Non Delinquent and 3,281. After I changed everything into numerical values by either using dummies or manulally chaning the values into numbers. 

## Testing and Balancing
For my models I tested diffrent Classification Regression models along with over and under sampling. Below is a chart showing the diffrent results I got for each model, I primarily looked at the recall score for each model because there is a high cost with a false negative. I decided to go with two models the first one is a prediction model (50/50) and the secound is a pridiction percentage model and I found the best threshold to use when I tuned the model later. I highlited the models Ichose for the final training and testing. 
### Model 1
<img src="Model_plots/Table1.png" height="300">

### Model 2
<img src="Model_plots/Table12.png" height="100>


## Tuning and Removing Features
After Choosing my models I 

![picture](Model_plots/Feature_Importances_RFP.png)

![picture](Model_plots/Feature_Importances_RF.png)

## Results
<img src="Model_plots/conf.png" width="700" height="700">
<img src = "Model_plots/prob_conf.png". width="700" height="700">

 
