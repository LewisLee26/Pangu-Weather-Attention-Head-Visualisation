## Summary
This web app is for interactively visualising the attention heads of the weather forecasting machine learning model, Pangu-Weather.

## Set Up

### Create environment
```bash
conda env create -f requirements.yml
conda activate pangu-weather-att-head-vis
```

### Download Model
Please download the four pre-trained models (~1.1GB each) from Google drive or Baidu netdisk:
- The 1-hour model (pangu_weather_1.onnx): [Google drive](https://drive.google.com/file/d/1fg5jkiN_5dHzKb-5H9Aw4MOmfILmeY-S/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1M7SAigVsCSH8hpw6DE8TDQ?pwd=ie0h)
- The 3-hour model (pangu_weather_3.onnx): [Google drive](https://drive.google.com/file/d/1EdoLlAXqE9iZLt9Ej9i-JW9LTJ9Jtewt/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/197fZsoiCqZYzKwM7tyRrfg?pwd=gmcl)
- The 6-hour model (pangu_weather_6.onnx): [Google drive](https://drive.google.com/file/d/1a4XTktkZa5GCtjQxDJb_fNaqTAUiEJu4/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/1q7IB7tNjqIwoGC7KVMPn4w?pwd=vxq3)
- The 24-hour model (pangu_weather_24.onnx): [Google drive](https://drive.google.com/file/d/1lweQlxcn9fG0zKNW8ne1Khr9ehRTI6HP/view?usp=share_link)/[Baidu netdisk](https://pan.baidu.com/s/179q2gkz2BrsOR6g3yfTVQg?pwd=eajy)


### Download Pangu Data
```bash
python download_data.py --start-date 2018-01-01 --end-date 2018-01-02
```

### Save Activations
```bash
python save_activations --model_num 24 --data-date 2018-01-01 --data-time 00:00 --intermediate_layers 0 1 2 4 --num_threads 4
```

### Compress Data
```bash
python compress_data.py
```
