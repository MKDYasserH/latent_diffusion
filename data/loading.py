import abc
import os

class BaseDataLoader(abc.ABC):
    def __init__(self, data_dir):
        self.data_dir = data_dir
    
    
    