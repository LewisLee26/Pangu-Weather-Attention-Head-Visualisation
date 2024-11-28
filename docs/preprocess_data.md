## Set Up
The `preprocess_data.py` script has two required arguments: `start_date` and `end_date`. It will download and preprocess data for the specified dates (at 00:00 and 12:00). It also accepts the following arguments:
- `model_num`: The forecast step of the model in hours (24, 6, 3, 1).
- `intermediate`: The index of the attention layers from which to obtain the output.
- `num_threads`: The number of threads to use when running the model.

```bash
python preprocess_data.py --start_date 2018-01-01 --end_date 2018-01-5 --model_num 24 --intermediate 0 1 2 3 --num_threads 4
```

The preprocess_data.py script runs three main sub-scripts: download_data.py, save_activations.py, and format_data.py. All of these scripts can be run can be run individually.

### Download Pangu Data
Run the script to download the Pangu data from [WeatherBench2](https://github.com/google-research/weatherbench2). The dates range from 2018 to 2022 and there are two times per day, 00:00 and 12:00.

```bash
python scripts/download_data.py --start_date 2018-01-01 --end_date 2018-01-02
```

### Save Activations
Save the attention patterns and outputs from the model using the downloaded data. The model can take multiple indexes for the intermediate_layers, it ranges from 0 to 11, corresponding to the 12 attention layers in the model.

```bash
python scripts/save_activations.py --model_num 24 --data_date 2018-01-01 --data_time 00:00 --intermediate_layers 0 1 2 3 --num_threads 4
```

### Format Data
Format the input and attention data to work for the web app visualisations. 

```bash
python scripts/format_data.py --data_date 2018-01-01 --data_time 00:00 --intermediate_layers 0 1 2 3 
```
