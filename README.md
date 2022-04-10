# Credit Default Classification System (description build in progress)

![](https://github.com/deelipvenkat/credit_default_classification_system/blob/main/project_demo.gif)

### OBJECTIVE
The goal of this project is to assist Banks/NBFC in the process of credit decision (i.e whether an application is approved or rejected) by working as a pre-filter for processing hundreds or thousands of applications received everyday so that the manual loan approval processes high quality loans. Also the number of features which a machine learning model considers might be impossible for a human to consider.


### ABOUT THE DATASET

I have used [PKDD'99](https://relational.fit.cvut.cz/dataset/Financial) Financial Dataset which is a real anonymized Czech Republic Bank data in ***relational database*** format.The bank has provided data about their clients, the accounts (transactions within several months), the loans already granted, the credit cards issued etc.


### TECHNOLOGY STACK USED 

I have built this project in jupyter notebook(python 3.8). ***MYSQL Connector*** for python was used for connecting to the database & converting the relational database into usuable format for building the machine learning model.I have used python flask to build the web application. Heroku Cloud Platform was used for the deployment of the model. To install all the dependencies for this project, download the requirements.txt file & run the below command line in the terminal.

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

### TRANSFORMING THE RELATIONAL DATASET INTO FINAL FORM 

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

### EXPLORATORY DATA ANALYSIS

We have a inital feature set of . Now let's perform our initial investigation of the data to find patterns & to check some of our assumptions based on our intution using statistical techniques & graphical representations. We have used tableau due to the speed & comfort it provides for performing EDA on datasets with large number of features.
Scipy was also used to perform some statisical tests.



### FEATURE SELECTION



### HANDLING MISSING VALUES
```
dt['status']=dt['status'].replace({'A':1,'B':0,'C':1,'D':0})


```


### ENCODING CATEGORICAL VARIABLES
```
dt['negative_balance']=dt['negative_balance'].fillna(0)
dt['issued']=dt['issued'].fillna(0)

```


### HANDLING OUTLIERS

```
dt.loc[(dt.negative_balance>0),'negative_balance']=1
dt.loc[(dt.issued != 0) ,'issued']=1
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

### MODEL TRAINING 





K-Fold Cross Validation is not suitable for handling imbalanced data because it randomly divides the data into k-folds. Folds might have negligible or no data from the minority class resulting in a highly biased results.To resolve this issue we use stratified k-fold which splits the data randomly & ensures that class imbalance distribution is maintained across each fold.

Also since our dataset is very small using k-fold cross validation will give us a better idea of how different algorithms are performing rather than a single test train split.

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

Since the dataset has class imbalance & the dataset size is small , it is better to stick to less complex models , as more complex models like random forest tend to overfit in such cases as we have seen in our tests above. So from the final results of the performance of various machine learning mdoels , it is wise to choose logistic regression as it tends to provide stable results & works well for class imbalances. 





### CREATING PICKLE FILES
Our final logistic regression model is stored in a pickle file which is used in deployment.

```

# The ml model & standard scaler is dumped into a pickle file.
filename = 'credit_model.pkl'
pickle.dump(classifier_lg, open(filename, 'wb'))
pickle.dump(sc, open('scaler.pkl', 'wb'))
```

### CREATING WEB APP USING FLASK

Now we create a new python file to build a web app & start importing necessary libraries. Please check the app.py file to see the complete flask application build.
```
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
```
We then create a flask application instance by passing '__name__' as argument to the flask class. The __name__ variable takes the value of the python source file we are using
In this case  it is 'app'.

```
app = Flask(__name__)
```
We then load our pickle files to use them in our deployment
```
model = pickle.load(open('credit_model.pkl', 'rb'))
scaler =pickle.load(open('scaler.pkl', 'rb'))
```


```
@app.route('/')
def home():
    return render_template('template_credit.html')
```
We want our webapp to take in inputs of values of all the features , run it throgh the ml model & return a respose whether the loan is approved or rejected.
```
@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        '''
    For rendering results on HTML GUI
    '''
        int_features = [float(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        final_features=scaler.transform(final_features)
    prediction = model.predict(final_features)
    if prediction == 1:
        
        output="LOAN APPROVED"
    else:
        output='LOAN APPLICATION REJECTED'
    

    return render_template('template_credit.html', prediction_text= output)


if __name__ == "__main__":
    app.run(debug=True)
```

### FRONT END BUILD OF THE APPLICATION

All the features inputs and the predict button & the placeholder for output of the ml model are built here. Since this is a fairly simple html file with comments written in the document. Please take a look for yourself.

```
<!DOCTYPE html>
<html >
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">	
  <title>CREDIT DECISION SYSTEM</title>
</head>
<body>
 <div class="login">
	<h1>AUTOMATED CREDIT DECISION SYSTEM</h1>

     <!-- Inputs of all the features which are then passed into our ml model are created here -->
     
    <form action="{{ url_for('predict')}}"method="post">
    	<input type="text" name="INSTALLMENTS" placeholder="INSTALLMENTS" required="required" />
        <input type="text" name="CARD_ISSUED" placeholder="CARD_ISSUED" required="required" />
		<input type="text" name="NO_CLIENTS" placeholder="NO_CLIENTS" required="required" />
        <input type="text" name="NEGATIVE_BALANCE" placeholder="NEGATIVE_BALANCE" required="required" />
                                      <input type="text" name="FIXED_SALARY" placeholder="FIXED_SALARY" required="required" />
        <button type="submit" class="btn btn-primary btn-block btn-large">predict</button>
    </form>
   <br>
   <br>
<!--The output of the ml model is connected here >	 
   {{ prediction_text }} 
 </div>
</body>
</html>
```

### DEPLOYING THE MODEL IN HEROKU


With creating the flask python file & application web template file. We have to create a requirements.txt file which contains all the libraries with the versions used for building the model.

Create a file name called Procfile (without .txt extension). A Procfile tells heroku how to start your application on it's servers.
```
web: gunicorn app:app
```
Web indicates that it is a web app. 
Gunicorn tells heroku to use gunicorn for your application.Gunicorn is a Python HTTP server for WSGI applications. It allows you to run Python applications concurrently by running multiple Python processes within a single dyno. Since it is a standard to use gunicorn in flask applications for deployments we are going to add it to our procfile. 
The 1st app represents the python file that runs your application(which is app.py in this case) & the 2nd app represents the app name.

Now with all the necessary elements available we are going to deploy our model into heroku cloud platform. We are not going to go in-depth here how to deploy the model in heroku. Please refer to this link here to learn how to deploy applications in heroku.

Since we have used python 3.8 in our local environment , we will specify heroku to use the same version so that there won't be any compatibility issues during deployment.

Create a new file in your root directory name runtime.txt and mention your python version in the exact format.
```
python-3.8.13
```

### DEPLOYMENT SAMPLES(add images)



### END NOTE 



### REFERENCES


