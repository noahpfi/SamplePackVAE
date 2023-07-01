import os

import util
from model import IAFVAEModel


class SamplePackGenerator:
    def __init__(self,
                 log_dir,
                 batch_size=1,
                 lib_dir=None):

        self.log_dir = log_dir
        self.batch_size = batch_size
        self.lib_dir = lib_dir

        # Check if a directory exists
        if not os.path.exists(self.log_dir):
            raise IOError(f'Directory {self.log_dir} not found. Train a model first.')

        # Check if parameters exist
        if os.path.isfile(f'{self.log_dir}/params.json'):
            print(f'Loading existing parameters: {self.log_dir}/params.json')
            self.params = util.load_params(f'{self.log_dir}/params.json')
        else:
            raise IOError(f'Parameters file {self.log_dir}/params.json not found. Train a model first.')

        # Create model
        print(f'Creating model...', end='')
        self.model = IAFVAEModel(self.params, keep_prob=1.0)
        print('Done')
















