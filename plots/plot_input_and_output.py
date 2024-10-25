import os 
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Data directories and paths
input_data_dir = 'input_data'
output_data_dir = 'output_data'
data_date = "2018-01-01"
data_time = "00:00"
input_name = "input_surface"
output_name = "output_surface"

# Load input data
input_path = os.path.join(input_data_dir, data_date, data_time, f"{input_name.replace('/', '_')}.npy")
input_data = np.load(input_path)  # Load all four surface variables (shape: 4, 721, 1440)
input_data = input_data[3]  # Assuming the 4th variable (index 3) is the one of interest

# Load output data
output_path = os.path.join(output_data_dir, data_date, data_time, f"{output_name.replace('/', '_')}.npy")
output_data = np.load(output_path)  # Load all four surface variables (shape: 4, 721, 1440)
output_data = output_data[3]  # Assuming the 4th variable (index 3) is the one of interest

# Set up longitudes and latitudes
lons = np.linspace(0, 360, 1440)
lons_shifted = np.where(lons > 180, lons - 360, lons)
lats = np.linspace(90, -90, 721)

# Create the figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot input data on the first subplot
ax = axs[0]
ax.coastlines()
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
im = ax.pcolormesh(lons_shifted, lats, input_data, cmap='turbo', transform=ccrs.PlateCarree())
plt.colorbar(im, ax=ax, orientation='horizontal', label='T2M Input')
ax.set_title('Input Surface Data')

# Plot output data on the second subplot
ax = axs[1]
ax.coastlines()
ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
im = ax.pcolormesh(lons_shifted, lats, output_data, cmap='turbo', transform=ccrs.PlateCarree())
plt.colorbar(im, ax=ax, orientation='horizontal', label='T2M Output')
ax.set_title('Output Surface Data')

# Add gridlines to both plots
window_lat_size = 24
window_lon_size = 48
lon_grid_lines = np.arange(-180, 180.1, 360 / (1440/window_lon_size))  # Longitudes divided into 48 sections
lat_grid_lines = np.arange(-90, 90.1, 180 / (720/window_lat_size))  # Latitudes divided into 24 sections

for ax in axs:
    gridlines = ax.gridlines(xlocs=lon_grid_lines, ylocs=lat_grid_lines, color='black', linestyle='--', draw_labels=True)
    gridlines.right_labels = False
    gridlines.top_labels = False

# Set a common title for the figure
plt.suptitle('T2M Input and Output Surface Variables Visualization with 24x48 Gridlines')

# Show the plot
plt.tight_layout()
plt.show()
