#Design a pipeline for use on the Melbourne housing dataset

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection as ms
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


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

#Extract numerical column NAMES
numerical_cols = list(X.select_dtypes(include = ['number']).columns.values)
categorical_cols = list(X.select_dtypes(include = ['object']).columns.values)

#Separate the data into training and validations sets
random_seed = 1234;
X_train,X_valid,y_train,y_valid = ms.train_test_split(X,y,test_size = 0.25, random_state = random_seed)

#Print some of the training and validation sets
print(X_train.head())
print(y_train.head())

#Create the pipeline

#Preprocess numerical data
numerical_transformer = SimpleImputer(strategy = 'constant')

#Preprocess categorical data
categorical_transformer = Pipeline(steps = [('imputer',SimpleImputer(strategy='most_frequent')),
('onehot',OneHotEncoder(handle_unknown = 'ignore'))])

#Bundle the transformers
preprocessor = ColumnTransformer( transformers = [('num',numerical_transformer,numerical_cols), ('cat',categorical_transformer,categorical_cols)] )

#Builld a model
model = RandomForestRegressor(n_estimators = 100, random_state = random_seed)

#Build pipeline
my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])

#Preprocess and fit model simultaneously
my_pipeline.fit(X_train, y_train)

#Use the pipeline to make predictions from validation data
preds = my_pipeline.predict(X_valid)

#Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE: ',score)
