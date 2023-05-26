import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'white')
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import accuracy_score

df = pd.read_csv('telco.csv')
df.drop('customerID',axis = 1, inplace=True)
df.head()

cat = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity',
      'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling',
       'PaymentMethod','Churn']

le = LabelEncoder()
for i in cat:
    df[i] = le.fit_transform(df[i])

df.head()

df.info()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors = 'coerce')
df.dtypes

df.isna().sum()

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
df.isna().sum()

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[num_cols].describe()

ss = StandardScaler()
df[num_cols] = ss.fit_transform(df[num_cols])
df.head()

X = df.drop('Churn',axis = 1).values
y = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 123, stratify = y)

rf = RandomForestClassifier()

kf = KFold(n_splits = 5, shuffle = True, random_state = 123)

rf.fit(X_train,y_train)
cv_scores_rf = cross_val_score(rf,X,y,scoring = 'accuracy')
print(np.mean(cv_scores_rf))

y_pred_rf = rf.predict(X_test)

print(confusion_matrix(y_test,y_pred_rf))

print(classification_report(y_test,y_pred_rf))

import pickle

filename = 'final_model.sav'
pickle.dump(rf,open(filename,'wb'))
