import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

# Sample traffic data (you should replace this with your actual data)
data = """Timestamp,Latitude,Longitude,Road ID,Speed (km/h),Traffic Flow,Vehicle Count,Weather Conditions,Incident Reports,Day of Week,Hour of Day
07-09-2024 08:00,40.74172214,-116.2316406,R8,23.95374551,Medium,813,Rain,None,Saturday,8
07-09-2024 08:15,40.742, -116.232,R1,30,High,1200,Clear,None,Saturday,8
07-09-2024 08:30,40.743,-116.233,R2,25,Low,400,Cloudy,None,Saturday,8
07-09-2024 08:45,40.744,-116.234,R3,28,Medium,800,Rain,Accident,Saturday,8
07-09-2024 09:00,40.745,-116.235,R4,35,High,1500,Snow,None,Saturday,9
08-09-2024 09:00,40.746,-116.236,R5,27,Medium,700,Rain,None,Sunday,9
09-09-2024 09:00,40.747,-116.237,R6,20,Low,300,Clear,Incident,Monday,9
10-09-2024 09:00,40.748,-116.238,R7,22,Medium,650,Cloudy,None,Tuesday,9
11-09-2024 10:00,40.749,-116.239,R8,19,High,1200,Rain,None,Wednesday,10
11-09-2024 11:00,40.740,-116.230,R9,33,Medium,900,Clear,None,Wednesday,11"""  # Add the full data string here

# Read the data into a DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data), parse_dates=['Timestamp'])

# Data Preprocessing
df['Traffic Flow'] = df['Traffic Flow'].astype('category')
df['Weather Conditions'] = df['Weather Conditions'].astype('category')

# Create a GeoDataFrame
geometry = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
geo_df = gpd.GeoDataFrame(df, geometry=geometry)

# Set the coordinate reference system (CRS)
geo_df.set_crs(epsg=4326, inplace=True)  # WGS84 Latitude/Longitude

# Basic statistics
print("Basic Statistics:")
print(geo_df.describe())

# Plotting traffic flow based on vehicle count
plt.figure(figsize=(12, 8))
base = geo_df.plot(column='Vehicle Count', cmap='OrRd', markersize=50, alpha=0.6, legend=True)
plt.title('Traffic Flow Visualization', fontsize=15)
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)

# Adding annotations
for x, y, label in zip(geo_df.geometry.x, geo_df.geometry.y, geo_df['Road ID']):
    plt.annotate(label, xy=(x, y), fontsize=8, ha='right')

plt.show()

# Analyze traffic flow by day of the week
traffic_by_day = geo_df.groupby('Day of Week')['Vehicle Count'].sum().sort_values(ascending=False)
print("\nTraffic Flow by Day of the Week:")
print(traffic_by_day)

# Plot traffic flow by day of the week
traffic_by_day.plot(kind='bar', color='skyblue', figsize=(10, 6))
plt.title('Total Vehicle Count by Day of the Week', fontsize=15)
plt.xlabel('Day of the Week', fontsize=12)
plt.ylabel('Total Vehicle Count', fontsize=12)
plt.xticks(rotation=45)
plt.show()

# Save the GeoDataFrame to a shapefile for further GIS analysis
output_shapefile = "traffic_data.shp"
geo_df.to_file(output_shapefile)
print(f"\nGeoDataFrame saved to {output_shapefile}")

# Analyze traffic speed distribution
plt.figure(figsize=(10, 6))
plt.hist(geo_df['Speed (km/h)'], bins=10, color='orange', edgecolor='black')
plt.title('Distribution of Traffic Speed', fontsize=15)
plt.xlabel('Speed (km/h)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid()
plt.show()

# Correlation heatmap
correlation_matrix = geo_df[['Speed (km/h)', 'Vehicle Count']].corr()
plt.figure(figsize=(6, 5))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.title('Correlation Heatmap', fontsize=15)
plt.xticks([0, 1], ['Speed', 'Vehicle Count'])
plt.yticks([0, 1], ['Speed', 'Vehicle Count'])
plt.show()

# Example of spatial join if you have a roads GeoDataFrame
# roads = gpd.read_file('roads.shp')  # Load road network
# joined = gpd.sjoin(geo_df, roads, how='inner', op='intersects')
# print("\nJoined GeoDataFrame with Roads:")
# print(joined.head())

