#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Bradley Gram-Hansen
Time created:  19:56
Date created:  26/04/2018

License: MIT
'''

import sys
import os

class experiment(dict):

    def __init__(self, *args, **kwargs):
        # {batch_size : int
        #  iterations : int
        #  initial_lr; float
        #   lr_decay : float
        #  flow_length: int
        #  name : str}
        # the following allows us to call the "keys" of the dict og the class as if they were attributes
        super(experiment, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def register_directory(self,name):
        """ Creates a directory to save experiment data"""
        PATH= sys.path[0]
        SAVE_PATH = os.path.join(PATH,name)
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)
        self.__setattr__(name, SAVE_PATH)

    def __setattr__(self, name, value):
        self.__dict__[name] = value





