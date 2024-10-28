import os
import numpy as np
import onnx
import onnxruntime as ort
from time import time
import argparse

def log_time(func):
    """Decorator to log the time taken by a function."""
    def wrapper(*args, **kwargs):
        start_time = time()
        print(f"{func.__name__.replace('_', ' ').title()}: Processing", end='\r')
        result = func(*args, **kwargs)
        end_time = time()
        print(f"{func.__name__.replace('_', ' ').title()}: Complete - Time Taken: {end_time - start_time:.2f}")
        return result
    return wrapper

@log_time
def load_model(path):
    """Load the ONNX model."""
    return onnx.load(path)

@log_time
def create_inter_output(model, layer_name_list):
    """Add intermediate layer output to the ONNX model."""
    value_info_protos = []
    shape_info = onnx.shape_inference.infer_shapes(model)
    for _, node in enumerate(shape_info.graph.value_info):
        if node.name in layer_name_list:
            value_info_protos.append(node)

    assert len(value_info_protos) == len(layer_name_list)
    model.graph.output.extend(value_info_protos)

    return model

@log_time
def create_session(model_path, output_names, num_threads):
    """Create an ONNX Runtime session."""
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = num_threads
    session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    
    session_outputs = session.get_outputs()
    output_names.extend([output.name for output in session_outputs])
    
    return session

@log_time
def load_data(input_dir, data_date, data_time):
    """Load input data."""
    input_upper = np.load(os.path.join(input_dir, data_date, data_time, 'input_upper.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_dir, data_date, data_time, 'input_surface.npy')).astype(np.float32)
    return input_upper, input_surface

@log_time
def run_model(session, input_data, input_surface_data, output_names):
    """Run the model inference."""
    return session.run(output_names, {'input': input_data, 'input_surface': input_surface_data})

@log_time
def save_model(model, path):
    """Save the modified ONNX model."""
    onnx.save(model, path)

@log_time
def save_output(output_data_dir, data_date, data_time, outputs, output_names):
    """Save the output data."""
    full_output_dir = os.path.join(output_data_dir, data_date, data_time)
    os.makedirs(full_output_dir, exist_ok=True)
    for i, output in enumerate(outputs):
        np.save(os.path.join(full_output_dir, f"{output_names[i].replace('/', '_')}.npy"), output)

def print_layer_names(model):
    """Print all layer names in the ONNX model."""
    for node in model.graph.node:
        print(node.name)

def print_layer_outputs(model):
    """Print all layer names in the ONNX model."""
    for node in model.graph.node:
        print(node.output)

def main(args):
    model_path = os.path.join(args.models_dir, f'pangu_weather_{args.model_num}.onnx')
    model = load_model(model_path)
    
    intermediate_layer_name = [
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

    selected_layers = [intermediate_layer_name[i] for i in args.intermediate_layers]
    model = create_inter_output(model, selected_layers)

    try:
        onnx.checker.check_model(model)
        print("Model Modifications: Valid")
    except onnx.checker.ValidationError as e:
        print(f"Model Modifications: Invalid - {e}")
        return

    modified_model_path = os.path.join(args.models_dir, f'pangu_weather_modified.onnx')
    save_model(model, modified_model_path)
    
    output_names = []
    ort_session = create_session(modified_model_path, output_names, args.num_threads)
    input_data, input_surface_data = load_data(args.input_data_dir, args.data_date, args.data_time)
    
    outputs = run_model(ort_session, input_data, input_surface_data, output_names)
    
    for i, output in enumerate(outputs):
        print(f"{output_names[i].title()} Shape:", output.shape)

    save_output(args.output_data_dir, args.data_date, args.data_time, outputs, output_names)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ONNX model with specified parameters.')
    parser.add_argument('--model_num', type=int, required=True, help='Model number to use.')
    parser.add_argument('--data_date', type=str, required=True, help='Date of the data in YYYY-MM-DD format.')
    parser.add_argument('--data_time', type=str, required=True, help='Time of the data in HH:MM format.')
    parser.add_argument('--intermediate_layers', type=int, nargs='+', required=True, help='Indices of intermediate layers to use.')
    parser.add_argument('--input_data_dir', type=str, default='input_data', help='Directory for input data.')
    parser.add_argument('--output_data_dir', type=str, default='output_data', help='Directory for output data.')
    parser.add_argument('--models_dir', type=str, default='checkpoints', help='Directory for model checkpoints.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use for ONNX Runtime session.')

    args = parser.parse_args()
    main(args)
