import pickle
from tqdm import tqdm
import os
import sys


from FlareAug import main

def to_augment():
    with open('to_augment_v3.p', 'rb') as f:
        to_augment = pickle.load(f)

    for augment in tqdm(to_augment):
        name = augment[0][1]
        main('./train/' + name, augment[1][0]['bbox'], f'./train_super/{sys.argv[0]}_' + name, 1)
        # main('./train/' + name, augment[1][0]['bbox'], './train_aug2/s_' + name, 2)



def check():
    path = './cars'
    for name in os.listdir(path):
        f = os.path.join(path, name)
        main(f, 0, './check/' + name, 1)

print(sys.argv[1])
to_augment()