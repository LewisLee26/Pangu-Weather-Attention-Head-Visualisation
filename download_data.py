import xarray as xr
import numpy as np
import os
from tqdm import trange
from time import time

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
def load_dataset():
    return xr.open_zarr('gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr')

# Prepare the directory path function
def prepare_directory(date_str, time_str):
    dir_path = f"input_data/{date_str}/{time_str}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path

def main():
    ds = load_dataset()

    start_date = '2018-01-01'
    end_date = '2018-01-31'

    # Select the dataset between start_date and end_date
    ds_filtered = ds.sel(time=slice(start_date, end_date))

    progress_bar = trange(len(ds_filtered['time']), desc="Processing time steps")

    # Process each time step
    for i in progress_bar:
        # Get the current time step
        current_time = ds_filtered['time'].isel(time=i).values
        date_str = np.datetime_as_string(current_time, unit='D')
        time_str = np.datetime_as_string(current_time, unit='m')[-5:]

        progress_bar.set_description(f"Saving Data: {date_str} {time_str}")

        # Create directory for the current time step
        dir_path = prepare_directory(date_str, time_str)

        # Extract and stack surface variables
        surface_vars = ['mean_sea_level_pressure', '10m_u_component_of_wind', 
                        '10m_v_component_of_wind', '2m_temperature']
        input_surface = np.stack([ds_filtered[var].isel(time=i, prediction_timedelta=0).values.astype(np.float32) 
                                  for var in surface_vars], axis=0)

        # Save input_surface.npy
        np.save(os.path.join(dir_path, "input_surface.npy"), input_surface)

        # Extract and stack upper-air variables
        upper_vars = ['geopotential', 'specific_humidity', 'temperature', 
                      'u_component_of_wind', 'v_component_of_wind']
        input_upper = np.stack([ds_filtered[var].isel(time=i, prediction_timedelta=0).values.astype(np.float32) 
                                for var in upper_vars], axis=0)

        # Save input_upper.npy
        np.save(os.path.join(dir_path, "input_upper.npy"), input_upper)

if __name__ == "__main__":
    main()
