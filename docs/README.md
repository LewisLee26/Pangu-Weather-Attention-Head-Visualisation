## Summary
This web app is for interactively visualising the attention heads of the weather forecasting machine learning model, [Pangu-Weather](https://github.com/198808xc/Pangu-Weather).

## Set Up

### Download Model
Please download the pre-trained models (~1.1GB each) from Google drive or Baidu netdisk:
- The 1-hour model (pangu_weather_1.onnx): [Google drive](https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1M7SAigVsCSH8hpw6DE8TDQ?pwd=ie0h)
- The 3-hour model (pangu_weather_3.onnx): [Google drive](https://drive.google.com/file/d/1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/197fZsoiCqZYzKwM7tyRrfg?pwd=gmcl)
- The 6-hour model (pangu_weather_6.onnx): [Google drive](https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3)
- The 24-hour model (pangu_weather_24.onnx): [Google drive](https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy)

### Create environment
This project uses [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to manage the Python packages. Run the following bash commands to set up and activate the Conda enviroment used for the data preparation Python scripts listed below. The Conda environment is not required for hosting the web app.

```bash
conda env create -f requirements.yml
conda activate pangu-weather-att-head-vis
```

### Preprocess Data
With the environment activated, run the preprocess_data.py script which downloads the pangu data, runs the models to extract the intermidate activations, and formats the data into binaries for the web app. 

Read more about how to set up the data at [preprocess_data.md](/docs/preprocess_data.md).

```bash
python scripts/preprocess_data.py --start_date 2018-01-01 --end_date 2018-01-05
```

### Host Web App
The web app can be hosted locally for easy access once all of the data has been formated correctly. By default, the web app can be access from [http://localhost:8000/main.html](http://localhost:8000/main.html).

```bash
python -m http.server --directory src
```
