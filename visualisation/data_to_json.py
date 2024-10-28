import os
import numpy as np
import json
from tqdm import trange

# Load your data
input_data_dir = '../input_data'
output_data_dir = '../output_data'
json_dir = 'json'
os.makedirs(json_dir, exist_ok=True)

data_date = "2018-01-01"
data_time = "00:00"
input_surface_name = "input_surface"
input_upper_name = "input_surface"
layer_name = "/b1/Add_14_output_0"

input_surface_path = os.path.join(input_data_dir, data_date, data_time, f"{input_surface_name.replace('/', '_')}.npy")
input_upper_path = os.path.join(input_data_dir, data_date, data_time, f"{input_surface_name.replace('/', '_')}.npy")
attention_path = os.path.join(output_data_dir, data_date, data_time, f"{layer_name.replace('/', '_')}.npy")

input_surface = np.load(input_surface_path)
input_upper = np.load(input_upper_path)
attention_output = np.load(attention_path)


# Define chunk sizes
chunk_size_lat = 24
chunk_size_lon = 48

# Split input_surface into chunks
for lat_index in trange(0, input_surface.shape[1], chunk_size_lat):
    for lon_index in range(0, input_surface.shape[2], chunk_size_lon):
        lat_end = min(lat_index + chunk_size_lat, input_surface.shape[1])
        lon_end = min(lon_index + chunk_size_lon, input_surface.shape[2])
        chunk_data = input_surface[:, lat_index:lat_end, lon_index:lon_end]
        chunk_filename = f"input_surface_{lat_index}_{lon_index}.json"
        with open(os.path.join(json_dir, chunk_filename), 'w') as f:
            json.dump(chunk_data.tolist(), f)

# Split attention_output into separate files for each window and head
for win_lon in trange(attention_output.shape[0]):
    for win_lat_pl in range(attention_output.shape[1]):
        for head in range(attention_output.shape[2]):
            attention_filtered = attention_output[win_lon, win_lat_pl, head, :, :]
            
            attention_filename = f"attention_{win_lon}_{win_lat_pl}_{head}.json"
            with open(os.path.join(json_dir, attention_filename), 'w') as f:
                json.dump(attention_filtered.tolist(), f)
