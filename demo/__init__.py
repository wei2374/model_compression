import os
import configparser
config = configparser.ConfigParser()
p = os.path.abspath('.')
config_path = os.path.join(p,'demo/config.cfg')
config.read(config_path)

model_dir = config['Model']['model_folder']
dataset_dir = config['Datasets']['dataset_folder']