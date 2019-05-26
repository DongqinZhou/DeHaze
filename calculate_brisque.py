# -*- coding: utf-8 -*-
'''
BRISQUE: https://ieeexplore.ieee.org/document/6272356 & https://www.learnopencv.com/image-quality-assessment-brisque/

Based on pybrisque: https://github.com/bukalapak/pybrisque

Installation of libsvm for python: https://stackoverflow.com/questions/12877167/how-do-i-install-libsvm-for-python-under-windows-7%7D

Make sure to set working directory to where brisque.py is stored.
'''
import os
import numpy as np
import brisque

path = ''
files = os.listdir(path)
BRI = []

bri = BRISQUE()

for file in files:
    BRI.append(bri.get_score(path + '/' + file))

print('Mean of BRISQUE is ', np.mean(BRI))
