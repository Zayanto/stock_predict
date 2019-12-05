import quandl
import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import cross_validate,train_test_split



from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm





company=(input(str("enter company name:")))

df = quandl.get("WIKI/"+company.upper())
print (company.upper())


#print(df.tail())

df = df[['Adj. Close']]

forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:] # set X_forecast equal to last 30
X = X[:-forecast_out] # remove last 30 from X

y = np.array(df['Prediction'])
y = y[:-forecast_out]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#print(df.tail())

# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("Next 30 days predicted prices: \n")#confidence)

forecast_prediction = clf.predict(X_forecast)
#print(forecast_prediction)
x=1
for i in forecast_prediction:
    
    date =  datetime.datetime.today().date() + datetime.timedelta(days=x)
    print(date,'=',i)
    x=x+1
