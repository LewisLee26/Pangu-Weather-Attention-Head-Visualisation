import os
import json
import numpy as np

# Load your data
input_data_dir = '../input_data'
output_data_dir = '../output_data'
bin_dir = 'bin'

data_date = "2018-01-01"
data_time = "00:00"
input_surface_name = "input_surface"
input_upper_name = "input_upper"

intermediate_layer_names = [
    '/b1/Add_output_0', 
    '/b1/Add_3_output_0',
    '/b1/Add_7_output_0', 
    '/b1/Add_10_output_0',
    '/b1/Add_14_output_0', 
    '/b1/Add_17_output_0',
    '/b1/Add_21_output_0', 
    '/b1/Add_24_output_0',
    '/b1/Add_28_output_0', 
    '/b1/Add_31_output_0',
    '/b1/Add_35_output_0', 
    '/b1/Add_38_output_0',
    '/b1/Add_42_output_0', 
    '/b1/Add_45_output_0',
    '/b1/Add_49_output_0',
    '/b1/Add_52_output_0',
]

# Function to save map data for a given configuration
def save_map_data(input_data, chunk_size_lat, chunk_size_lon, config_name, data_type, roll_data=False):
    if roll_data:
        # Roll the data by half the size of the window
        input_data = np.roll(input_data, shift=(chunk_size_lat // 2, chunk_size_lon // 2), axis=(1, 2))
    
    for lat_index in range(0, input_data.shape[1], chunk_size_lat):
        for lon_index in range(0, input_data.shape[2], chunk_size_lon):
            lat_end = min(lat_index + chunk_size_lat, input_data.shape[1])
            lon_end = min(lon_index + chunk_size_lon, input_data.shape[2])
            chunk_data = input_data[:, lat_index:lat_end, lon_index:lon_end]
            
            # Create directory structure
            map_dir = os.path.join(bin_dir, data_date, data_time, config_name, 'map')
            os.makedirs(map_dir, exist_ok=True)
            
            chunk_filename = f"{data_type}_{lat_index}_{lon_index}.bin"
            chunk_data.tofile(os.path.join(map_dir, chunk_filename))

# Load input surface data
input_surface_path = os.path.join(input_data_dir, data_date, data_time, f"{input_surface_name.replace('/', '_')}.npy")
input_surface = np.load(input_surface_path)

# Save map data for input surface
save_map_data(input_surface, 24, 48, 'config_24x48', 'input_surface')
save_map_data(input_surface, 48, 96, 'config_48x96', 'input_surface')

# Save rolled map data for input surface
save_map_data(input_surface, 24, 48, 'config_24x48_shifted', 'input_surface', roll_data=True)
save_map_data(input_surface, 48, 96, 'config_48x96_shifted', 'input_surface', roll_data=True)

# Load input upper data
input_upper_path = os.path.join(input_data_dir, data_date, data_time, f"{input_upper_name.replace('/', '_')}.npy")
input_upper = np.load(input_upper_path)

# Split input upper data into 13 lots of (5, 721, 1440)
for i in range(input_upper.shape[1]):
    upper_data_chunk = input_upper[:, i, :, :]
    save_map_data(upper_data_chunk, 24, 48, f'config_24x48_upper_{i}', 'input_upper')
    save_map_data(upper_data_chunk, 48, 96, f'config_48x96_upper_{i}', 'input_upper')
    
    # Save rolled map data for input upper
    save_map_data(upper_data_chunk, 24, 48, f'config_24x48_upper_{i}_shifted', 'input_upper', roll_data=True)
    save_map_data(upper_data_chunk, 48, 96, f'config_48x96_upper_{i}_shifted', 'input_upper', roll_data=True)

# Process attention data per layer
for layer_index, layer_name in enumerate(intermediate_layer_names):
    layer_name_safe = layer_name.replace('/', '_')
    attention_path = os.path.join(output_data_dir, data_date, data_time, f"{layer_name_safe}.npy")
    attention_output = np.load(attention_path)

    # Determine the number of heads and window size based on layer index
    if layer_index < 2 or layer_index >= len(intermediate_layer_names) - 2:
        num_heads = 6
    else:
        num_heads = 12

    # Split and save attention_output in binary format
    for lon in range(attention_output.shape[0]):
        for lat_pl in range(attention_output.shape[1]):
            for head in range(num_heads):
                attention_chunk = attention_output[lon, lat_pl, head, :, :]
                
                # Create directory structure
                attention_dir = os.path.join(bin_dir, data_date, data_time, layer_name_safe, 'attention')
                os.makedirs(attention_dir, exist_ok=True)
                
                attention_filename = f"attention_{lon}_{lat_pl}_{head}.bin"
                attention_chunk.tofile(os.path.join(attention_dir, attention_filename))

# Generate JSON file with available dates, times, and layers (excluding config maps)
available_data = {}

for date in os.listdir(bin_dir):
    date_path = os.path.join(bin_dir, date)
    if os.path.isdir(date_path):
        available_data[date] = {}
        for time in os.listdir(date_path):
            time_path = os.path.join(date_path, time)
            if os.path.isdir(time_path):
                available_data[date][time] = []
                for layer in os.listdir(time_path):
                    layer_path = os.path.join(time_path, layer)
                    if os.path.isdir(layer_path) and 'config' not in layer:
                        available_data[date][time].append(layer)

with open('available_data.json', 'w') as json_file:
    json.dump(available_data, json_file, indent=4)
