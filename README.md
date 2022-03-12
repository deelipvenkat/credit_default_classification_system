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



