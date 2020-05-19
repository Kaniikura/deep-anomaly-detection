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
``` bash
$ python tools/make_csvs.py
```

# Training
In configs directory, you can find configurations I used to produce my models.

## Train models
To train models, run following commands.

``` bash
$ python run.py train_gan with {config_path} -f
```
