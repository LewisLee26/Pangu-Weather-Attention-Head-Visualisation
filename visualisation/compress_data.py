import os
import numpy as np

# Load your data
input_data_dir = '../input_data'
output_data_dir = '../output_data'

data_date = "2018-01-01"
data_time = "00:00"
input_surface_name = "input_surface"

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

layer_name = intermediate_layer_names[0]

input_surface_path = os.path.join(input_data_dir, data_date, data_time, f"{input_surface_name.replace('/', '_')}.npy")
attention_path = os.path.join(output_data_dir, data_date, data_time, f"{layer_name.replace('/', '_')}.npy")

input_surface = np.load(input_surface_path)
attention_output = np.load(attention_path)

# Chunk the input surface data
chunk_size_lat = 24
chunk_size_lon = 48

for lat_index in range(0, input_surface.shape[1], chunk_size_lat):
    for lon_index in range(0, input_surface.shape[2], chunk_size_lon):
        lat_end = min(lat_index + chunk_size_lat, input_surface.shape[1])
        lon_end = min(lon_index + chunk_size_lon, input_surface.shape[2])
        chunk_data = input_surface[:, lat_index:lat_end, lon_index:lon_end]
        
        # Create directory structure
        map_dir = os.path.join('bin', data_date, data_time, 'map')
        os.makedirs(map_dir, exist_ok=True)
        
        chunk_filename = f"input_surface_{lat_index}_{lon_index}.bin"
        chunk_data.tofile(os.path.join(map_dir, chunk_filename))

# Split and save attention_output in binary format
for lon in range(attention_output.shape[0]):
    for lat_pl in range(attention_output.shape[1]):
        for head in range(attention_output.shape[2]):
            attention_chunk = attention_output[lon, lat_pl, head, :, :]
            
            # Create directory structure
            attention_dir = os.path.join('bin', data_date, data_time, 'attention')
            os.makedirs(attention_dir, exist_ok=True)
            
            attention_filename = f"attention_{lon}_{lat_pl}_{head}.bin"
            attention_chunk.tofile(os.path.join(attention_dir, attention_filename))
