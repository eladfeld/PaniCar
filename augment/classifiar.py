import numpy as np
import cv2 as cv
import os
import csv
from tqdm import tqdm

test = False
dir_path = './train/'
file = open(dir_path + '../pred_train.csv', 'w')
writer = csv.writer(file, lineterminator='\n')


for img in (tqdm(os.listdir(dir_path)) if not test else os.listdir(dir_path)):
    frame = cv.imread(dir_path + img)
    mean = np.mean(frame.ravel())
    if test:
        print(mean)
    raw = []
    if mean > 60:
        if test:
            print('day')
        raw = [img, 'day']
    else:
        if test:
            print('night')
        raw = [img, 'night']

    writer.writerow(raw)
    if test:
        cv.imshow('ff', frame)
        cv.waitKey(0)