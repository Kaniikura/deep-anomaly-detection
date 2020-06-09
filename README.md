# deep-anomaly-detection

# Set Environment
``` bash
$ conda env create -f myenv.yaml
```

# Prepare dataset
## Download dataset
Download and extract CIFAR10, MNIST and MVTec Anomaly Detection Dataset.

``` bash
$ python tools/create_data_folder.py
```

## Generate CSV files
Create the dataframes needed to pass data to the model
``` bash
$ python tools/make_csvs.py
```

# Training
In configs directory, you can find configurations I used to produce my models.

## Train models
To train models, run following commands.

``` bash
$ python run{_gan, _autoencoder, _metric_learning}.py train with {config_path} -f
```

## Inferecne
``` bash
$ python run{_gan, _autoencoder, _metric_learning}.py inference with {config_path} -f
```
