import json
import csv
from random import sample

import numpy as np
import pickle
classifications = []
with open('train_class/classifications.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        classifications.append(row)


cla = np.array(classifications)[:, 1]
nights = cla == 'night'
night_images = np.array(classifications)[nights]
images_names = night_images[:, 0]

data = []
with open('coco_format_train.json', 'r') as file:
    data = json.load(file)

indexes =data['images']

indexes = [(x['id'], x['file_name']) for x in indexes if x['file_name'] in images_names]
#
#
# annotations = data['annotations']
#
# ids = [id[0] for id in indexes]
# annotations = [x for x in annotations if x['image_id'] in ids]
#
#
#
# annotations_dict = {}
# for index in ids:
#     annotations_dict[index] = []


#
with open('annotations_dict.p', 'rb') as f:
    annotations_dict = pickle.load(f)

with open('annotations.p', 'rb') as f:
    annotations = pickle.load(f)

# for annotation in annotations:
#     if annotation['category_id'] == 3 or annotation['category_id'] == 4:
#         annotations_dict[annotation['image_id']].append(annotation)
#
#
# with open('annotations_dict.p', 'wb') as f:
#     pickle.dump(annotations_dict, f)


# annotations_per_image = {}
# for index in annotations_dict:
#     annotations_per_image[index] = len(annotations_dict[index])

candidates = sample(indexes, int(1 * len(indexes)))
print(len(candidates))

to_augment = []

for candidate in candidates:

    annotations_ = annotations_dict[candidate[0]]
    if len(annotations_) > 0:
        annotate = sample(annotations_, 1)
        to_augment.append((candidate, annotate))


with open('to_augment_v3.p', 'wb') as f:
    pickle.dump(to_augment, f)
