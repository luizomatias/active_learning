import os.path
import sys

def set_root_folder():
    HERE = os.path.dirname(os.path.abspath(__file__))
    UP_DIR = '/'.join(HERE.split('/')[0:-1])
    sys.path = [UP_DIR] + sys.path

set_root_folder()