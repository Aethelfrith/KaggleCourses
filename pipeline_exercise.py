#Make a pipeline for training a model predicting housing prices

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import BaseEstimator, TransformerMixin


#From https://stackoverflow.com/questions/48743032/get-intermediate-data-state-in-scikit-learn-pipeline
#Define a class such that data can be accessed at an intermediate step in the pipeline
#Inherit from BaseEstimator and TransformerMixin to comply with the requirements of a step in the pipeline
class StateGetter(BaseEstimator, TransformerMixin):
	
	def transform(self, X):
		#Make the shape accessible
		self.X = X
		return X
	
	def fit(self, X, y = None, **fit_params):
		return self
		
#Make a function which extracts the column names (in order??) of the preprocessed data

def get_cols_from_ct_pipeline(pipeline,preprocessor_name = 'preprocessor'):
	#Get all columnn names from the column transformer part of a pipeline
	#Check that the pipeline indeed has a preprocessor step of type ColumnTransformer
	if not   isinstance(pipeline.named_steps[preprocessor_name],ColumnTransformer):
		throw: TypeError("Type must be ColumnTransformer")
	
	transformers = pipeline.named_steps["preprocessor"].transformers
	cols = []
	#all_columns = transformers[0][2] + transformers[1][2] 
	all_columns = [cols + (transformers[i][2]) for i in range(len(transformers))] 
	return all_columns
	

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
#numerical_transformer = SimpleImputer(strategy='constant')
numerical_transformer = SimpleImputer(strategy='mean')

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
n_estimators = 10
model = RandomForestRegressor(n_estimators = n_estimators,random_state = rnd_seed)

#Bundle preprocessing and modelling
pipel = Pipeline(steps = [('preprocessor',preprocessor),('stategetter',StateGetter()), ('model',model)])

#Make predictions with the pipeline
pipel.fit(X_train,y_train)

#Preprocess and predict validation data
preds = pipel.predict(X_valid)

print("MAE: ",mean_absolute_error(y_valid,preds))

#Check which features are the most important
feature_importances = pipel.named_steps.model.feature_importances_
#feature_names = pipel.named_steps["preprocessor"].transformers[0][2]

feature_names = get_cols_from_ct_pipeline(pipel)

print(feature_names)

#Make a Series with the feature names
#feature_importances_series = pd.Series(data = feature_importances, index = X_train.columns.values)

#print('X_train_columns: ',X_train.columns.values)

#print('Feature importances: ', feature_importances)
#print(feature_importances_series)
#Visualise the feature impportances in a barplot

#plt.figure(figsize = (10,7))
#sns.distplot()
#plt.show()

