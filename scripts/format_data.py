import os
import json
import numpy as np
import argparse
from tqdm import tqdm, trange
import shutil

# Define the intermediate layer names
INTERMEDIATE_LAYER_NAMES = [
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

def clear_directory(directory, verbose=False):
    """Remove all files and subdirectories in the specified directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    if verbose:
        print(f"Cleared directory: {directory}")

def save_map_data(input_data, chunk_size_lat, chunk_size_lon, config_name, data_type, roll_data=False, verbose=False):
    if roll_data:
        input_data = np.roll(input_data, shift=(chunk_size_lat // 2, chunk_size_lon // 2), axis=(-2, -1))
    
    for lat_index in range(0, input_data.shape[1], chunk_size_lat):
        for lon_index in range(0, input_data.shape[2], chunk_size_lon):
            lat_end = min(lat_index + chunk_size_lat, input_data.shape[1])
            lon_end = min(lon_index + chunk_size_lon, input_data.shape[2])
            chunk_data = input_data[:, lat_index:lat_end, lon_index:lon_end]
            
            map_dir = os.path.join(args.src_dir, 'bin', args.data_date, args.data_time, config_name, 'map')
            os.makedirs(map_dir, exist_ok=True)
            
            chunk_filename = f"{data_type}_{lat_index}_{lon_index}.bin"
            chunk_data.tofile(os.path.join(map_dir, chunk_filename))
            if verbose:
                print(f"Saved chunk: {chunk_filename}")

def main(args):
    # Clear the bin directory before processing
    bin_dir = os.path.join(args.src_dir, 'bin', args.data_date, args.data_time)
    clear_directory(bin_dir, verbose=args.verbose)

    input_surface_path = os.path.join(args.input_data_dir, args.data_date, args.data_time, f"{args.input_surface_name.replace('/', '_')}.npy")
    input_surface = np.load(input_surface_path)

    save_map_data(input_surface, 24, 48, 'config_24x48', 'input_surface', verbose=args.verbose)
    save_map_data(input_surface, 48, 96, 'config_48x96', 'input_surface', verbose=args.verbose)
    save_map_data(input_surface, 24, 48, 'config_24x48_shifted', 'input_surface', roll_data=True, verbose=args.verbose)
    save_map_data(input_surface, 48, 96, 'config_48x96_shifted', 'input_surface', roll_data=True, verbose=args.verbose)

    input_upper_path = os.path.join(args.input_data_dir, args.data_date, args.data_time, f"{args.input_upper_name.replace('/', '_')}.npy")
    input_upper = np.load(input_upper_path)

    for i in trange(input_upper.shape[1], desc="Processing upper data chunks", disable=not args.verbose):
        upper_data_chunk = input_upper[:, i, :, :]
        save_map_data(upper_data_chunk, 24, 48, f'config_24x48_upper_{i}', 'input_upper', verbose=args.verbose)
        save_map_data(upper_data_chunk, 48, 96, f'config_48x96_upper_{i}', 'input_upper', verbose=args.verbose)
        save_map_data(upper_data_chunk, 24, 48, f'config_24x48_upper_{i}_shifted', 'input_upper', roll_data=True, verbose=args.verbose)
        save_map_data(upper_data_chunk, 48, 96, f'config_48x96_upper_{i}_shifted', 'input_upper', roll_data=True, verbose=args.verbose)

    for layer_index in tqdm(args.intermediate_layers, desc="Processing intermediate layers", disable=not args.verbose):
        if layer_index < 0 or layer_index >= len(INTERMEDIATE_LAYER_NAMES):
            print(f"Invalid layer index: {layer_index}. Skipping.")
            continue

        layer_name = INTERMEDIATE_LAYER_NAMES[layer_index]
        layer_name_safe = layer_name.replace('/', '_')
        attention_path = os.path.join(args.output_data_dir, args.data_date, args.data_time, f"{layer_name_safe}.npy")
        
        if not os.path.exists(attention_path):
            print(f"Attention data not found for layer: {layer_name_safe}. Skipping.")
            continue

        attention_output = np.load(attention_path)

        num_heads = 6 if layer_index < 2 or layer_index >= len(INTERMEDIATE_LAYER_NAMES) - 2 else 12

        for lon in range(attention_output.shape[0]):
            for lat_pl in range(attention_output.shape[1]):
                for head in range(num_heads):
                    attention_chunk = attention_output[lon, lat_pl, head, :, :]
                    attention_dir = os.path.join(args.src_dir, 'bin', args.data_date, args.data_time, layer_name_safe, 'attention')
                    os.makedirs(attention_dir, exist_ok=True)
                    attention_filename = f"attention_{lon}_{lat_pl}_{head}.bin"
                    attention_chunk.tofile(os.path.join(attention_dir, attention_filename))
                    if args.verbose:
                        print(f"Saved attention chunk: {attention_filename}")

    available_data = {}
    for date in os.listdir(os.path.join(args.src_dir, 'bin')):
        date_path = os.path.join(args.src_dir, 'bin', date)
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

    json_dir = os.path.join(args.src_dir, 'available_data.json')
    with open(json_dir, 'w') as json_file:
        json.dump(available_data, json_file, indent=4)
    if args.verbose:
        print(f"Saved available data to {json_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ONNX model outputs and save map data.')
    parser.add_argument('--data_date', type=str, required=True, help='Date of the data in YYYY-MM-DD format.')
    parser.add_argument('--data_time', type=str, required=True, help='Time of the data in HH:MM format.')
    parser.add_argument('--intermediate_layers', type=int, nargs='+', required=True, help='Indices of intermediate layers to process.')
    parser.add_argument('--input_data_dir', type=str, default='input_data', help='Directory for input data.')
    parser.add_argument('--output_data_dir', type=str, default='output_data', help='Directory for output data.')
    parser.add_argument('--src_dir', type=str, default='src', help='Directory for binary output.')
    parser.add_argument('--input_surface_name', type=str, default='input_surface', help='Name of the input surface file.')
    parser.add_argument('--input_upper_name', type=str, default='input_upper', help='Name of the input upper file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')

    args = parser.parse_args()
    main(args)
