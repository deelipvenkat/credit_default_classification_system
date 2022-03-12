# Automated_credit_decision_system

### OBJECTIVE
The goal of this project is to assist Banks/NBFC in the process of credit decision (i.e whether an application is approved or rejected) by working as a pre-filter for processing hundreds or thousands of applications received everyday so that the manual loan approval processes high quality loans. Also the number of features which a machine learning model considers might be impossible for a human to consider.


### ABOUT THE DATASET

I have used [PKDD'99](https://relational.fit.cvut.cz/dataset/Financial) Financial Dataset which is a real anonymized Czech Republic Bank data in relational database format.The bank has provided data about their clients, the accounts (transactions within several months), the loans already granted, the credit cards issued etc.


### TECHNOLOGY STACK USED 

I have built this project in jupyter notebook(python 3.7). MYSQL Connector for python was used for connecting to the database & converting the relational database into usuable format for building the machine learning model.I have used python flask to build the web application. Heroku Cloud Platform was used for the deployment of the model. To install all the dependencies for this project, download the requirements.txt file & run the below command line in the terminal.

```
pip install -r requirements.txt
```

### IMPORTING LIBRARIES

```
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mysql import connector
import pickle
```

### CONNECTING TO DATABASE USING MYSQL/CONNECTOR 

```
mydb = connector.connect(                # credentials for the PKDD'99 database are given in the dataset link shared above 
  host="relational.fit.cvut.cz",
  user="guest",
  password="relational",
  database="financial"  
)
```

### MUNGING THE DATA IN FINAL STRUCTURED FORM 

```
# LOAN FEATURE TRACING WITH OWNER CLIENT ID & DETAILS

q=SELECT * FROM loan 
     JOIN account ON loan.account_id=account.account_id 
     JOIN disp ON account.account_id=disp.account_id
     JOIN client ON disp.client_id=client.client_id
     LEFT JOIN card ON disp.disp_id=card.disp_id
     LEFT JOIN district ON client.district_id=district.district_id
     where disp.type="OWNER";
    
    
    
```






### SMOTE FOR BALANCING DATASET

```
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=0)
x_train, y_train = sm.fit_resample(x_train, y_train)

```

### FEATURE SCALING

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
```

### VARIANCE INFLATION FACTOR (for multi-collinearity check)
```
# variance inflation factor to check multi-collinearity between features.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = fe.columns
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x_train, i)
						for i in range(len(fe.columns))]
```


### IMPLEMENTATION OF STRATIFIED K-FOLD
```
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
steps = list()
steps.append(('scaler', StandardScaler()))
steps.append(('log_model', LogisticRegression(random_state=0,C=0.00001)))
pipeline = Pipeline(steps=steps)
# define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
# evaluate the model using cross-validation
scores = cross_val_score(pipeline, x_train, y_train, scoring='f1', cv=cv, n_jobs=-1)
# report performance
print('f1_score: mean: {} , std :{}'.format(scores.mean()*100, scores.std()*100))
```

f1_score: mean: 81.3485323661658 , std :3.627310607374865

### CREATING PICKLE FILES
```
# The ml model & standard scaler is dumped into a pickle file.
filename = 'credit_model.pkl'
pickle.dump(classifier_lg, open(filename, 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))
```
