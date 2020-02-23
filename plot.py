from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloaderFP as dtl
import regressionalgorithmsFP as algs
import matplotlib.pyplot as plt

y = [3.7810761836304305, 3.4372058437839814, 3.6202080868569109, 3.7321257587367249]
x = [0.1,0.05,0.01,0.005]
plt.plot(x,y)
#plt.xticks([0.1,0.05,0.01,0.005,0.001])
plt.xlabel('stepsize')
plt.ylabel('everage error')
#plt.legend()
plt.show()