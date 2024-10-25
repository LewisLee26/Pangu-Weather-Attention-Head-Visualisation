import os 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Data directories and paths
input_data_dir = 'input_data'
data_date = "2018-01-01"
data_time = "00:00"
input_name = "input_surface"

path = os.path.join(input_data_dir, data_date, data_time, f"{input_name.replace('/', '_')}.npy")
data = np.load(path)  # Load all four surface variables (shape: 4, 721, 1440)

data = data[3]

# Create the map projection
fig = plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add coastlines and set extent
ax.coastlines()
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

# Plot data with correct extent
lons = np.linspace(0, 360, 1440)
lons_shifted = np.where(lons > 180, lons - 360, lons)
lats = np.linspace(90, -90, 721)

im = ax.pcolormesh(lons_shifted, lats, data, cmap='turbo', transform=ccrs.PlateCarree())
plt.colorbar(im, label='T2M')


window_lat_size = 24
window_lon_size = 48

# Add gridlines to match the 24x48 division
lon_grid_lines = np.arange(-180, 180.1, 360 / (1440/window_lon_size))  # Longitudes divided into 48 sections (7.5° each)
lat_grid_lines = np.arange(-90, 90.1, 180 / (720/window_lat_size))  # Latitudes divided into 24 sections (7.5° each)

# Add gridlines to the map with appropriate spacing
gridlines = ax.gridlines(xlocs=lon_grid_lines, ylocs=lat_grid_lines, color='black', linestyle='--', draw_labels=True)
gridlines.right_labels = False
gridlines.top_labels = False

plt.title('T2M Surface Variable Visualization with 24x48 Gridlines')
plt.show()


