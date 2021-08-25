# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 13:18:23 2016
used on  Wen May  12 13:18:23 2021

@author: steven
@modified: chh
"""
import matplotlib.pyplot as plt
import numpy as np
lines = np.loadtxt("episode_reward.txt", comments="#", delimiter="\n", unpack=False)

plt.plot(lines)
plt.show()