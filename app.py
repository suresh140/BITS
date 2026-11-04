# Shree Ganeshaya Namagh
# Roll No
# Name : Suresh Sharma
# Assignment No 2
# Subject : Machine Learning
# Using Scikit-learn for modelling
#1. Logistic Regression
#2. Decision Tree Classifier
#3. K-Nearest Neighbor Classifier
#4. Naive Bayes Classifier - Gaussian or Multinomial
#5. Support Vector Machine
import os

from pathlib import Path
import pickle
import warnings
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score,precision_score,recall_score,f1_score,matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

file_path = "spambase.csv"
TARGET = "spam"             # your target column (binary: 0/1)
SEED = 42
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
df=kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS,"colormap/spambase",file_path)
print(df.head(10))
print(df.columns.to_list())
print(df.shape)  # check for number of rows and features
print(df["spam"].value_counts())
print(df.isna().sum().sum())
n_spam = (df["spam"] == 1).sum()
no_spam = (df["spam"] == 0).sum()
print(f'No of spam emails is {n_spam}')
print(f'No of non spam emails is {no_spam}')
print(f' checking for the consistency {(n_spam+no_spam)/df.shape[0]}')


# Spliting the data into training and testing sets

X = df.drop(columns=["spam"])             # features
y = df["spam"]                            # target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% for testing
    random_state=42,      # fixed seed so the split is reproducible
    stratify=y            # keep the class proportions the same in both sets
)

print(X_train.shape, X_test.shape)

# Training a logistic regression model

model1 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEED)),
])

model2 = DecisionTreeClassifier(max_depth=8,random_state=SEED)

model3 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier(n_neighbors=5)),
])

model4 = GaussianNB()

model5 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf",probability=True,random_state=SEED)),
])

Models = {"lr":model1,"dtree":model2,"knn":model3,"naivebaiyes":model4,"svm":model5}


#-------Function for Model learning and saving as pickle
def fit_model(models1):
    for name in models1:
        models1[name].fit(X_train,y_train)
        str = "model/"+name+".pkl"
        with open(str, "wb") as f:
            pickle.dump(models1[name], f)

fit_model(Models)

#-----Function for evaluating the models
def evaluate(models2):
    output = []
    for name in models2:
        y_pred = models2[name].predict(X_test)
        y_prob = models2[name].predict_proba(X_test)[:, 1]  # P(class 1), used for AUC

        accuracy =  round(accuracy_score(y_test, y_pred), 4)
        AUC = float(roc_auc_score(y_test, y_prob))
        Precision = round(precision_score(y_test, y_pred), 4)
        Recall = round(recall_score(y_test, y_pred), 4)
        F1=round(f1_score(y_test, y_pred), 4)
        MCC=float(matthews_corrcoef(y_test, y_pred))
        confusion=confusion_matrix(y_test, y_pred)
        output.append({"model_name":name,"accuracy":accuracy,"AUC":AUC,"precision":Precision,"Recall":Recall,
                  "f1":F1,"MCC":MCC,"Confusion":confusion})

        print(output)
    return output

        #print("\nClassification report:\n", classification_report(y_test, y_pred))
op1 = evaluate(Models)
df = pd.DataFrame(op1)
metric_df=df.drop(['Confusion'],axis=1)
df.to_csv("metrics.csv")

print(f'After function {op1}')
st.table(metric_df)

@st.cache_resource



def load_model(filename):
    with open(MODEL_DIR/filename, "rb") as f:
        return pickle.load(f)



models = {
    "Logistic Regression": load_model("lr.pkl"),
    "Decision Tree": load_model("dtree.pkl"),
    "KNN": load_model("knn.pkl"),
    "Naive Baiyes": load_model("naivebaiyes.pkl"),
    "SVM": load_model("svm.pkl"),
}

choice = st.selectbox("Choose model", list(models))

uploaded = st.file_uploader("Upload CSV for prediction", type="csv")
if uploaded:
    new_df = pd.read_csv(uploaded)
    preds = models[choice].predict(new_df)
    out = new_df.copy()
    out["Prediction"] = preds
    st.dataframe(out)

