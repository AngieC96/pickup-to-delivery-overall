import sys
import configparser


# Config parsing
def load_config(config_file='./config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)

    try:
        dwh_config = config['dwh']
        livedb_config = config['live_db']
        parameters_config = config['parameters']

    except Exception as e:
        print(f"Error while reading {config_file}, exception {e}")
        return None

    return dwh_config, livedb_config, parameters_config


# Query parameter preparation
def create_string_from_list(list):
    string = "', '"
    string = string.join(list)

    return string

def update_progress(progress):
    """ Displays or updates a console progress bar.
    Accepts a float between 0 and 1. Any int will be converted to a float.
    A value under 0 represents a 'halt'.
    A value at 1 or bigger represents 100%
    """
    barLength = 50 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress bar must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done!\r\n"
    block = int(round(barLength*progress))
    text = "\r[{0}] {1}% {2}".format( "#"*block + "."*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()