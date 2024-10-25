import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

output_data_dir = 'output_data'
data_date = "2018-01-01"
data_time = "00:00"
layer_name = "/b1/Add_14_output_0"

path = os.path.join(output_data_dir, data_date, data_time, f"{layer_name.replace("/", "_")}.npy")
attention_output = np.load(path)

head = 8
win_lon = 0
win_lat_pl = 0

attention_filtered = attention_output[win_lon, win_lat_pl, head, :, :]

# Create labels for each combination of pressure level, latitude, and longitude
pressure_levels = [0, 1]  # Example pressure levels
latitudes = [f"lat{i}" for i in range(6)]  # Example latitude labels
longitudes = [f"lon{j}" for j in range(12)]  # Example longitude labels

pl_colour = "red"
lat_colour = "blue"
lon_colour = "green"

labels = []
for p in pressure_levels:
    for lat in latitudes:
        for lon in longitudes:
            labels.append(f"P{p}_{lat}_{lon}")

# Plotting
plt.figure(figsize=(12, 10))
plt.imshow(attention_filtered, aspect='auto', cmap='viridis')  # Use a colormap of your choice
plt.colorbar()  # Optional: Add a colorbar to indicate values
plt.title('Attention Visualization')

# # Set the ticks and labels
# plt.xticks(ticks=np.arange(144), labels=labels, rotation=90, fontsize=8)
# plt.yticks(ticks=np.arange(144), labels=labels, fontsize=8)

# Add grid lines to separate pressure levels, latitudes, and longitudes
for i in range(1, len(pressure_levels)):
    plt.axhline(y=i*72-0.5, color=pl_colour, linestyle='--', linewidth=1)  # Horizontal line for pressure level
    plt.axvline(x=i*72-0.5, color=pl_colour, linestyle='--', linewidth=1)  # Vertical line for pressure level

for i in range(1, len(latitudes) * len(pressure_levels)):
    plt.axhline(y=i*12-0.5, color=lat_colour, linestyle='--', linewidth=0.5)  # Horizontal line for latitude
    plt.axvline(x=i*12-0.5, color=lat_colour, linestyle='--', linewidth=0.5)  # Vertical line for latitude

# Annotations for pressure levels
for i, pl in enumerate(pressure_levels):
    plt.text(-5, i*72 + 36, f'pl{pl}', color=pl_colour, fontsize=10, ha='right', va='center')
    plt.text(i*72 + 36, 150, f'pl{pl}', color=pl_colour, fontsize=10, ha='right', va='center')
    for j, lat in enumerate(latitudes):
        plt.text(-5, (i * 6*12) + j*12 + 6, f'{lat}', color=lat_colour, fontsize=10, ha='right', va='center')
        plt.text((i * 6*12) + j*12 + 6, 150, f'{lat}', color=lat_colour, fontsize=10, ha='right', va='center')

plt.tight_layout()
plt.show()
