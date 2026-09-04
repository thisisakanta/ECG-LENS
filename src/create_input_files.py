import os
import sys
from utils import create_decoder_input_files
import configparser

config = configparser.ConfigParser()
config.read(os.path.join("configs", "config_create_input_files.ini"))

# Print the contents of the config file
for section in config.sections():
    print(f'[{section}]')
    for key, value in config.items(section):
        print(f'{key} = {value}')
    print()


file_names_and_paths = config["file_names_and_paths"]

if __name__ == '__main__':
    print("############################################")
    print("ECG folder is " + file_names_and_paths["ecg_folder"])
    print("output folder is " + file_names_and_paths["output_folder"])
    print("############################################")

    create_decoder_input_files(
        data_folder=file_names_and_paths["ecg_folder"],
        output_folder=file_names_and_paths["output_folder"],
        sampling_rate=config.getint("file_creation_params", "sampling_rate"),
        replace_abbr_text=config.getboolean("file_creation_params", "replace_abbr_text"),
        min_word_freq=config.getint("file_creation_params", "min_word_freq"),
        vocab_size=config.getint("file_creation_params", "vocab_size"),
        max_len=config.getint("file_creation_params", "max_len"),
        translate_comments=config.getboolean("file_creation_params", "translate_comments"),
        split_method=config["file_creation_params"]["split_method"],
        debug_mode=config.getboolean("file_creation_params", "debug_mode")
        )
