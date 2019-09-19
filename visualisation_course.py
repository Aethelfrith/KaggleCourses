import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#PART 1
#Register datatime converters
#from pandas.plotting import register_matplotlib_converters
#register_matplotlib_converters()

##Load data
#fifa_filepath = "./Data/fifa.csv"
#fifa_data = pd.read_csv(fifa_filepath, index_col ="Date", parse_dates = True)

##Inspect first rows
#print(fifa_data.head())

#Make line chart
#plt.figure(figsize=(16,6))
#sns.lineplot(data=fifa_data)
#plt.show()


#PART 2
#Read the file
spotify_filename = "./Data/spotify.csv"
spotify_data = pd.read_csv(spotify_filename, index_col="Date",parse_dates=True)

#Inspect data
print(spotify_data.head())


