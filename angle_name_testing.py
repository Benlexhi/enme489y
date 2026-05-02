# ENME 489Y: Remote Sensing

# Import packages
import numpy as np
import argparse
import cv2
import imutils
import glob
import re
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open(r'C:\Users\Brian\PycharmProjects\enme489y\test_scans\testresults.txt') as f:
    for line in f:
        parts = line.split()
        if parts != ['0', '0', '0'] and float(parts[2]) != 0.0:
            print(line.strip())
            break

#files = glob.glob(r"C:\Users\Brian\PycharmProjects\enme489y\test_scans\*.jpg")
#for f in files:
#    angle = int(re.findall(r'\d+', f)[-1])
#    print(f, "->", angle)