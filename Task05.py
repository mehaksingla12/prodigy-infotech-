# Task-05: US Traffic Accident EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# 1. Load the dataset
df = pd.read_csv('US_Accidents_March23.csv')

# 2. Preprocessing
df['Start_Time'] = pd.to_datetime(df['Start_Time'])
df['Hour'] = df['Start_Time'].dt.hour
df['Weekday'] = df['Start_Time'].dt.day_name()

# 3. Plot accident count by hour
plt.figure(figsize=(10, 5))
sns.countplot(x='Hour', data=df, palette='magma')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Accident Count')
plt.grid(True)
plt.show()

# 4. Plot accident count by weather condition
plt.figure(figsize=(12, 5))
top_weather = df['Weather_Condition'].value_counts().nlargest(10)
sns.barplot(x=top_weather.index, y=top_weather.values, palette='viridis')
plt.title('Top 10 Weather Conditions During Accidents')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Plot accident count by road conditions
plt.figure(figsize=(10, 5))
top_conditions = df['Road_Condition'].value_counts().nlargest(5)
sns.barplot(x=top_conditions.index, y=top_conditions.values, palette='coolwarm')
plt.title('Top Road Conditions at Accident Time')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 6. Accidents by weekday
plt.figure(figsize=(10, 5))
sns.countplot(x='Weekday', data=df, order=[
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    palette='Set2')
plt.title('Accidents by Day of the Week')
plt.ylabel('Accident Count')
plt.show()

# 7. Heatmap of accident hotspots (sampled for performance)
sample_df = df[['Start_Lat', 'Start_Lng']].dropna().sample(n=10000, random_state=1)

# Generate map
m = folium.Map(location=[sample_df['Start_Lat'].mean(), sample_df['Start_Lng'].mean()], zoom_start=5)
HeatMap(sample_df.values, radius=8, blur=6).add_to(m)
m.save("accident_hotspots_map.html")
print("Heatmap saved as 'accident_hotspots_map.html'")
