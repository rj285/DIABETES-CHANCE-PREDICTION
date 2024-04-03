import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('health_data.csv')

df = pd.DataFrame(data)

# print(data.isna().sum()) 
"""
Glucose Level (mg/dL)                    2
BMI                                      1
Insulin Level (mu U/ml)                  2
Age (years)                              0
Likelihood of Developing Diabetes (%)    0
"""
median_glucose = math.floor(df['Glucose Level'].median())
# print(median_glucose)
df['Glucose Level'] = df['Glucose Level'].fillna(median_glucose)
# print(df['Glucose Level (mg/dL)'])

median_bmi = math.floor(df['BMI'].median())
# print(median_bmi)
df['BMI'] = df['BMI'].fillna(median_bmi)
# print(df['BMI'])

median_insulin = math.floor(df['Insulin Level'].median())
# print(median_insulin)
df['Insulin Level'] = df['Insulin Level'].fillna(median_insulin)
# print(df['Insulin Level (mu U/ml)'])


x = df[['Glucose Level','BMI','Insulin Level','Age']]
y = df['Likelihood of Developing Diabetes']

reg = LinearRegression()
reg.fit(x,y)

print("--- DIABETES CHANCE PREDICTION---")
print("Example: 110.0,25.0,6.0,35,15")

GL = float(input("GLUCOSE:- "))
BMI = float(input("BMI:- "))
IL = float(input("INSULINE:- "))
AGE = float(input("AGE:- "))

prediction = reg.predict([[GL,BMI,IL,AGE]])
print(prediction)

