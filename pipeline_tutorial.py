#Design a pipeline for use on the Melbourne housing dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms

#Load the data
housing_filename = "./Data/melb_data.csv"
housing_data = pd.read_csv(housing_filename)

#Inspect the data
print(housing_data.head())
print(housing_data.describe())

#Separate the target variable
y = housing_data["Price"]
housing_data.drop("Price",axis=1,inplace=True)
X =  housing_data.copy()

y_col = y.name
X_col = list(X.columns.values)
#print(y_col)
#print(X_col)

#Separate the data into training and validations sets
random_seed = 1234;
X_train,X_valid,y_train,y_valid = ms.train_test_split(X,y,test_size = 0.25, random_state = random_seed)

#Print some of the training and validation sets
print(X_train.head())
print(y_train.head())


