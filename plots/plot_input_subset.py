import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Data directories and paths
input_data_dir = 'input_data'
data_date = "2018-01-01"
data_time = "00:00"
input_name = "input_surface"

path = os.path.join(input_data_dir, data_date, data_time, f"{input_name.replace('/', '_')}.npy")
input_surface = np.load(path)  # Load all four surface variables (shape: 4, 721, 1440)

patch_size_lat = 4
patch_size_lon = 4

# Define the chunk size
chunk_size_lat = 24
chunk_size_lon = 48

# Define the starting indices for the subsection
lat_index = 6
lon_index = 29

# Calculate the end indices based on the chunk size
lat_start = lat_index * chunk_size_lat
lat_end = lat_start + chunk_size_lat
lon_start = lon_index * chunk_size_lon
lon_end = lon_start + chunk_size_lon

# Ensure indices do not exceed the data dimensions
lat_end = min(lat_end, input_surface.shape[1])
lon_end = min(lon_end, input_surface.shape[2])

surface_var_idx = {
    "MSLP": 0,
    "U10": 1,
    "V10": 2,
    "T2M": 3,
}

weather_var = "T2M"

# Extract the subsection for MSLP (Mean Sea Level Pressure)
chunk_data = input_surface[surface_var_idx[weather_var], lat_start:lat_end, lon_start:lon_end]

# Define the latitude and longitude ranges for the subsection
latitudes = np.linspace(90, -90, 721)[lat_start:lat_end]
longitudes = np.linspace(0, 359.75, 1440)[lon_start:lon_end]

# Define the grid spacing based on the original latitude and longitude
lat_spacing = np.abs(latitudes[1] - latitudes[0])
lon_spacing = np.abs(longitudes[1] - longitudes[0])

# Shift latitudes and longitudes by half a grid cell
lat_shifted = latitudes - lat_spacing / 2
lon_shifted = longitudes - lon_spacing / 2

# Create a meshgrid for plotting with the shifted lat/lon
lon_grid, lat_grid = np.meshgrid(lon_shifted, lat_shifted)

# Plot the data using Cartopy
plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([longitudes.min(), longitudes.max(), latitudes.min(), latitudes.max()], crs=ccrs.PlateCarree())

# Add coastlines for context
ax.add_feature(cfeature.COASTLINE)

# Plot the data using pcolormesh with the shifted grid
im = ax.pcolormesh(lon_grid, lat_grid, chunk_data, cmap='turbo', transform=ccrs.PlateCarree())

# Add a colorbar
plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, label=weather_var)

# Add gridlines based on patch size
for i in range(lat_start, lat_end, patch_size_lat):
    ax.plot([longitudes.min(), longitudes.max()], [latitudes[i - lat_start], latitudes[i - lat_start]], color='black', linewidth=0.5, transform=ccrs.PlateCarree())

for j in range(lon_start, lon_end, patch_size_lon):
    ax.plot([longitudes[j - lon_start], longitudes[j - lon_start]], [latitudes.min(), latitudes.max()], color='black', linewidth=0.5, transform=ccrs.PlateCarree())

# Show the plot
plt.title(weather_var)
plt.show()
