#Design a pipeline for use on the Melbourne housing dataset

import pandas as pd
import matplotlib.pyplot as plt

#Load the data
housing_filename = "./Data/melb_data.csv"
housing_data = pd.read_csv(housing_filename)

#Inspect tha data
print(housing_data.head())
print(housing_data.describe())
