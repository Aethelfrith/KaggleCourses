import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Register datatime converters
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

#PART 1

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
print(spotify_data.tail())

#Print names of columns
list(spotify_data.columns)

#Plot data
plt.figure(figsize=(14,6))
plt.title("Daily global streams of popular songs in 2017-2018")
plt.xlabel('Date')
sns.lineplot(data=spotify_data['Shape of You'],label='Shape of You')
sns.lineplot(data =spotify_data['Despacito'],label='Descpacito')
plt.show()





