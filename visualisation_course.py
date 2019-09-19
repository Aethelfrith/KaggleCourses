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
##Read the file
#spotify_filename = "./Data/spotify.csv"
#spotify_data = pd.read_csv(spotify_filename, index_col="Date",parse_dates=True)

##Inspect data
#print(spotify_data.head())
#print(spotify_data.tail())

##Print names of columns
#list(spotify_data.columns)

##Plot data
#plt.figure(figsize=(14,6))
#plt.title("Daily global streams of popular songs in 2017-2018")
#plt.xlabel('Date')
#sns.lineplot(data=spotify_data['Shape of You'],label='Shape of You')
#sns.lineplot(data =spotify_data['Despacito'],label='Descpacito')
#plt.show()

#PART 3
##Read file
#flight_filepath = "./Data/flight_delays.csv"
#flight_data = pd.read_csv(flight_filepath,index_col = "Month")

##Print all of the data
#print(flight_data)

##Plot the data for Spirit Airlines in a barplot
##plt.figure(figsize=(10,6))
##plt.title("Average Arrival Delay for Spirit Airlines Flights by Month")
##sns.barplot(x=flight_data.index,y=flight_data['NK'])
##plt.ylabel("Arrival delay (minutes)")
##plt.show()

##Make heatmap of delays across companies and months
#plt.figure(figsize=(14,7))
#plt.title("Avg delay per airline and month")
#ax=sns.heatmap(data=flight_data,annot=True)
#plt.xlabel("Airline")
##plt.ylabel("Month")
##Fix heatmap y limits
#y_lim = ax.get_ylim()
#new_y_lim = (y_lim[0] + 0.5, y_lim[1]-0.5)
#ax.set_ylim(new_y_lim)
#plt.show()

##PART 3 Exercise
##Read file
#ign_filepath = "./Data/ign_scores.csv"
#ign_data = pd.read_csv(ign_filepath, index_col="Platform")

##Print data
#print(ign_data)

##Find the maximum and minimum scores
##Find the maximum score by platform
#maximum_scores = ign_data.max(axis = 0)
#maximum_score = maximum_scores.max()
#print("Maximum scores across genre: \n")
#print(maximum_scores)
#print("Maximum scores across genre and platform: " + str(maximum_score))

###Make a bar plot of avg score among racing games
##plt.figure(figsize=(10,9))
##plt.title("Average score across racing games")
##avg_genre_scores_bplot = sns.barplot(x=ign_data.index,y = ign_data["Racing"])
###Rotate xticklabels
##avg_genre_scores_bplot.set_xticklabels(avg_genre_scores_bplot.get_xticklabels(),rotation=45)
##plt.show()

##Make a heatmap across games and genres
#plt.figure(figsize=(10,10))
#plt.title("Average scores across platform and genre")
#heatmap_handle = sns.heatmap(data=ign_data,annot=True)
#plt.xlabel("Platform")
#y_lim = heatmap_handle.get_ylim()
#new_y_lim = (y_lim[0] + 0.5, y_lim[1]-0.5)
#heatmap_handle.set_ylim(new_y_lim)
#plt.show()

#PART 5 Scatterplots
#insurance_filepath = "./Data/insurance.csv"
#insurance_data = pd.read_csv(insurance_filepath)

##Inspect the data
#print(insurance_data.head())
#print(insurance_data.describe())

##Scatterplot
##sns.scatterplot(x=insurance_data["bmi"],y = insurance_data['charges'],hue=insurance_data["smoker"])
##plt.show()

##sns.lmplot(x="bmi",y="charges",hue = "smoker",data=insurance_data)
##plt.show()

#sns.swarmplot(x=insurance_data["smoker"],y=insurance_data["charges"])
#plt.show()

##PART 5 Exercise
#candy_filepath = "./Data/candy.csv"
#candy_data = pd.read_csv(candy_filepath)

##Inspect
#print(candy_data.head())

##Inspect scatterplot of data
##sns.scatterplot(x=candy_data["sugarpercent"],y = candy_data["winpercent"], hue = candy_data["chocolate"])
##plt.show()

##sns.lmplot(x= "sugarpercent",y = "winpercent",hue = "chocolate",data = candy_data)
##plt.show()

#sns.swarmplot(x = candy_data["chocolate"],y = candy_data["winpercent"])
#plt.show()

#PART 6: Distributions
iris_filepath = "./Data/iris.csv"
iris_data = pd.read_csv(iris_filepath, col_index = "Id")

#Show data
print(iris_data.head())

