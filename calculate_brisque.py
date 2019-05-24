# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:59:32 2019

@author: Zero_Zhou
"""

import os
import numpy as np
import brisque

path = r'H:\Undergraduate\18-19-3\Undergraduate Thesis\Dataset\Test_RTTS Results\Real_MSCNN'
files = os.listdir(path)
BRI = []

bri = BRISQUE()

for file in files:
    BRI.append(bri.get_score(path + '/' + file))

print('Mean of BRISQUE is ', np.mean(BRI))
