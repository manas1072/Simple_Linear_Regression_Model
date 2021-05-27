import pandas as pd
import numpy as np
import sklearn
import joblib

data=pd.read_csv("/root/Salary_Data.csv")

X=data['YearsExperience']
X=np.array(X)
X=X.reshape(-1,1)

y=data['Salary']

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(X,y)
joblib.dump(model,'trained.pk1')
