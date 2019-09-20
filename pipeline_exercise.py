#Make a pipeline for training a model predicting housing prices

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#Set random seed
rnd_seed = 1234

#Read data
full_filename = "./Data/train_pipelines.csv"
test_full_filename = "./Data/test_pipelines.csv"
X_full = pd.read_csv(full_filename,index_col = 'Id')
X_test_full = pd.read_csv(test_full_filename, index_col = 'Id')

#Remove rows with missing target
X_full.dropna(axis=0,subset = ['SalePrice'],inplace=True)
#Separate target
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

#Separate validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size = 0.8, random_state = rnd_seed)

#Select categorical variables with small cardinality for onehot encoding, to avoid curse of dimensionality
max_cardinality_for_onehot = 10
categorical_cols = [colname for colname in X_train_full.columns if (X_train_full[colname].dtype == "object") and  (X_train_full[colname].nunique() < max_cardinality_for_onehot)]

#Select numerical columns
numerical_cols = list(X_train_full.select_dtypes(include = ['number']).columns.values)

#Only keep numerical and categorical columns
keep_cols = categorical_cols + numerical_cols
X_train = X_train_full[keep_cols].copy()
X_valid = X_valid_full[keep_cols].copy()
X_test = X_test_full[keep_cols].copy()

#Inspect data
print(X_train.head())

#Prepare preprocessing numerical data
numerical_transformer = SimpleImputer(strategy='constant')

#Prepare preprocessing categorical data
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(handle_unknown='ignore'))
])

#Bundle preprocessing patterns
preprocessor = ColumnTransformer(transformers = [
('num',numerical_transformer, numerical_cols),
('cat',categorical_transformer, categorical_cols)
])

#Define model
n_estimators = 100
model = RandomForestRegressor(n_estimators = n_estimators,random_state = rnd_seed)

#Bundle preprocessing and modelling
pipel = Pipeline(steps = [('preprocessor',preprocessor), ('model',model)])
