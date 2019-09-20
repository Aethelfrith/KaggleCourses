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
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
		
#Make a function which extracts the column names (in order??) of the preprocessed data
def get_cols_from_num_and_cat(X_train_full, num_cols, cat_cols, cat_transformer, onehot_name = 'onehot'):
	#Assume that there only onehot encoding changes the number of columns. Assume that the numerical columns are neither permuted nor changed on number.
	
	#Warning: This pythonic way of checking emptiness is borken by numpy arrays
	if not num_cols:
		num_columns = []
	else:
		num_dummy_training = X_train_full[num_cols].copy()
		num_columns = num_dummy_training.columns.values
	if not cat_cols:
		 cat_columns_onehot = []
	else:
		cat_dummy_training = X_train_full[cat_cols].copy()
		categorical_transformer.fit(cat_dummy_training)
		cat_columns_onehot = cat_transformer.named_steps['onehot'].get_feature_names(cat_cols)
	

	#The order of concatenation should be important! order should agree with order that of operations
	all_columns = list(num_columns) + list(cat_columns_onehot)
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
max_cardinality_for_onehot = 1
categorical_cols = [colname for colname in X_train_full.columns if (X_train_full[colname].dtype == "object") and (X_train_full[colname].nunique() < max_cardinality_for_onehot)]

#Select numerical columns
numerical_cols = list(X_train_full.select_dtypes(include = ['number']).columns.values)

##Only keep numerical and categorical columns
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
#model = DecisionTreeRegressor(random_state = rnd_seed)

#Bundle preprocessing and modelling
pipel = Pipeline(steps = [('preprocessor',preprocessor), ('model',model)])

#Make predictions with the pipeline
pipel.fit(X_train,y_train)

#Preprocess and predict validation data
preds = pipel.predict(X_valid)

print("MAE: ",mean_absolute_error(y_valid,preds))

#Check which features are the most important
feature_importances = pipel.named_steps.model.feature_importances_
#feature_names = pipel.named_steps["preprocessor"].transformers[0][2]


#To get the feature names, apply the onehotencoder on the original categorical data once more. Not pretty, but apparently the only way atm
print(categorical_cols)
feature_names = get_cols_from_num_and_cat(X_train_full, numerical_cols, categorical_cols, categorical_transformer)

print("Number of feature names: ",len(feature_names))
print("Number of feature importances: ",len(feature_importances))


#Make a Series with the feature names
feature_importances_series = pd.Series(data = feature_importances, index = feature_names)

#print(feature_importances_series)

#Visualise the feature impportances in a barplo
plt.figure(figsize = (10,7))
sns.distplot(feature_importances_series)
plt.show()

