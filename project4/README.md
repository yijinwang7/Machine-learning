Download dataset

[ESC-50](https://github.com/karolpiczak/ESC-50)

### Preprocessing

The preprocessing is done separately to save time during the training of the models.

For ESC-50: 
```console
python preprocessing/preprocessingESC.py --csv_file /path/to/esc50.csv --data_dir /path/to/audio_data/ --store_dir /path/to/store_spectrograms/ --sampling_rate 44100
```


### Training the Models

The configurations for training the models are provided in the config folder. The sample_config.json explains the details of all the variables in the configurations. The command for training is: 
```console
python train.py --config_path /config/your_config.json
```


###
Experiment one: change the num_fold in the configuration file to 5.

Experiment two: change the num_fold to 1. change the weight decay to 0

Experiment three: change pretrained to false

Experiment three: change data augmented to false


### Test

Run main.py , change the string1/string2/string3/string4 to the path of produced model checkpoint from the experiment

