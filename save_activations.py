import os
import numpy as np
import onnx
import onnxruntime as ort
from time import time

# Directories for input, output, and models
input_data_dir = 'input_data'
output_data_dir = 'output_data'
models_dir = 'checkpoints'
model_num = 24
data_date = '2018-01-07'
data_time = '00:00'
model_path = os.path.join(models_dir, f'pangu_weather_{model_num}.onnx')

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
def create_session(model_path, output_names):
    """Create an ONNX Runtime session."""
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(model_path, sess_options=options, providers=['CPUExecutionProvider'])
    
    # Set the outputs to include the intermediate layer
    session_outputs = session.get_outputs()
    output_names.extend([output.name for output in session_outputs])
    
    return session

@log_time
def load_data(input_dir):
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
    # Construct the full path for the output directory
    full_output_dir = os.path.join(output_data_dir, data_date, data_time)
    
    # Create the directory if it doesn't exist
    os.makedirs(full_output_dir, exist_ok=True)
    
    # Save each output
    for i, output in enumerate(outputs):
        np.save(os.path.join(full_output_dir, f"{output_names[i].replace("/", "_")}.npy"), output)

def print_layer_names(model):
    """Print all layer names in the ONNX model."""
    for node in model.graph.node:
        print(node.name)

def print_layer_outputs(model):
    """Print all layer names in the ONNX model."""
    for node in model.graph.node:
        print(node.output)


def main():
    model = load_model(model_path)
    
    intermediate_layer_name = [
        # Attention
        '/b1/Add_output_0', 
        # '/b1/Add_7_output_0', 
        # '/b1/Add_14_output_0', 
        # '/b1/Add_21_output_0', 
        # '/b1/Add_28_output_0', 
        # '/b1/Add_35_output_0', 
        # '/b1/Add_42_output_0', 
        # '/b1/Add_49_output_0',
        #
        # # Rolled Attention 
        # '/b1/Add_3_output_0',
        # '/b1/Add_10_output_0',
        # '/b1/Add_17_output_0',
        # '/b1/Add_24_output_0',
        # '/b1/Add_31_output_0',
        # '/b1/Add_38_output_0',
        # '/b1/Add_45_output_0',
        # '/b1/Add_52_output_0',
    ]

    model = create_inter_output(model, intermediate_layer_name)

    # Check the modified model for consistency
    try:
        onnx.checker.check_model(model)
        print("Model Modifications: Valid")
    except onnx.checker.ValidationError as e:
        print(f"Model Modifications: Invalid - {e}")
        return

    # Save the modified model
    modified_model_path = os.path.join(models_dir, f'pangu_weather_modified.onnx')
    save_model(model, modified_model_path)
    
    # Specify the intermediate layer name
    output_names = []
    
    ort_session = create_session(modified_model_path, output_names)
    input_data, input_surface_data = load_data(input_data_dir)
    
    # Run the model and get the intermediate layer output
    outputs = run_model(ort_session, input_data, input_surface_data, output_names)
    
    for i, output in enumerate(outputs):
        print(f"{output_names[i].title()} Shape:", output.shape)

    save_output(output_data_dir, data_date, data_time, outputs, output_names)
    
if __name__ == "__main__":
    start_time = time()
    main()
