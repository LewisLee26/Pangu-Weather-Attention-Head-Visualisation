import os
import numpy as np

# Load your data
input_data_dir = '../input_data'
output_data_dir = '../output_data'
bin_dir = 'bin'
os.makedirs(bin_dir, exist_ok=True)

data_date = "2018-01-01"
data_time = "00:00"
input_surface_name = "input_surface"
layer_name = "/b1/Add_output_0"

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
        chunk_filename = f"input_surface_{lat_index}_{lon_index}.bin"
        chunk_data.tofile(os.path.join(bin_dir, chunk_filename))

# Split and save attention_output in binary format
for lon in range(attention_output.shape[0]):
    for lat_pl in range(attention_output.shape[1]):
        for head in range(attention_output.shape[2]):
            attention_chunk = attention_output[lon, lat_pl, head, :, :]
            attention_filename = f"attention_{lon}_{lat_pl}_{head}.bin"
            attention_chunk.tofile(os.path.join(bin_dir, attention_filename))
