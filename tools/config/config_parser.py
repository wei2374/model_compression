import os
import configparser

config = configparser.ConfigParser()
p = os.path.abspath('.')
config_path = os.path.join(p,'demo/config.cfg')
config.read(config_path)

def get_model_dir():
    return config['Model']['model_folder']

def get_loss():
    return config['Training']['loss']

def get_dataset_format():
    return config['Datasets']['format']

def get_dataset_dir():
    return config['Datasets']['dataset_folder']