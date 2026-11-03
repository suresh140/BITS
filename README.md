Problem Statement:

Classification of emails into Spam and Non Spam based 
on the contents of the email

Dataset Description:

Dataset used is from Kaggle repository

Dataset name = Spambase

Its a single class classification model which classifies 
emails as spam or non spam

The dataset contains 58 columns with 57 features and one target

Its a big dataset containing 4601 records for training and testing purposes

The dataset does not have any anamoly and same is testing in the program


Github Link :


	
|model_name | accuracy | AUC  |precision |Recall |f1 |MCC|
|-|----------|------|----------|-------|---|------
|lr | 0.9294   |0.9702 |0.9209 |0.8981 |0.9093 |0.8518|
|
|dtree | 0.9034  |0.9154  |0.9029 |0.8457 |0.8734 |0.7965|
|
|knn | 0.9077  |0.9506  |0.8861  |0.8788 |0.8824 |0.8065|
|naivebaiyes | 0.8339  |0.9450  |0.7178  |0.9532 |0.8189 |0.6941|
|svm |0.9273  |0.9667  |0.9277  |0.8843  |0.9055 |0.8471|


## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---------------|----------------------|
| Logistic Regression | Highest accuracy along with high AUC |
| Decision Tree | Fast but low accuracy|
| KNN | need to define nearest neighbour count..slow for single class problem |
| Naive Bayes | “Accuracy 0.83 with highest recall 0.9532 fastest; |
| Support Vector Machine | Almost at par with the linear regression model|
| **Overall, Winner for your dataset?** | Logistic regression as it has all parameters good for a single|

