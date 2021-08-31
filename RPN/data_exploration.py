from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import json
from collections import Counter
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

data_directory = "data/"
annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "data/train/images"
TRAIN_ANNOTATIONS_PATH = "data/train/annotations.json"

VAL_IMAGES_DIRECTORY = "data/val/images"
VAL_ANNOTATIONS_PATH = "data/val/annotations.json"


coco = COCO(TRAIN_ANNOTATIONS_PATH)

category_ids = coco.loadCats(coco.getCatIds())
category_names = [_["name"] for _ in category_ids]
", ".join(category_names)
# print(category_names)

category_ids_list = [_["id"] for _ in category_ids]
# print(category_ids_list)

image_ids = coco.getImgIds()
# print(image_ids)

# random_image_id = random.choice(image_ids)
# img = coco.loadImgs(random_image_id)[0]
# image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img["file_name"])
# I = io.imread(image_path)
# plt.imshow(I)
'''
image_widths = []
image_heights = []
image_areas = []
annotation_areas = []
category_ids_list = []
crowds = []

for i in image_ids:
    img = coco.loadImgs(i)[0]
    image_widths.append(img["width"])
    image_heights.append(img["height"])
    image_areas.append(img["width"] * img["height"])
    annotation_ids = coco.getAnnIds(imgIds=img['id'])
    annotations = coco.loadAnns(annotation_ids)
    for a in annotations:
        annotation_areas.append(a["area"])
        category_ids_list.append(a["category_id"])
        crowds.append(a["iscrowd"])

save = {}
save["image_widths"] = image_widths
save["image_heights"] = image_heights
save["image_areas"] = image_areas
save["annotation_areas"] = annotation_areas
save["category_ids_list"] = category_ids_list
save["crowds"] = crowds

with open("data_informations.json", "w") as f:
    json.dump(save, f)

'''
f = open("data_informations.json")
save = json.load(f)
'''
# plt.hist([el for el in save["image_areas"] if (el < 2*1e6)], 50)
counter = Counter(save["category_ids_list"])
ids = list(counter.keys())
occurrences = list(counter.values())
data = zip(ids, occurrences)
data = sorted(data, key=lambda tup: tup[1], reverse=True) 
occurrences = [tup[1] for tup in data]
print(max(occurrences), min(occurrences))
# plt.bar(range(0,len(data)), occurrences)
# plt.xlabel("Category")
# plt.title("Categories")
# plt.show()
'''

counter = Counter(save["crowds"])
print(counter)
print(len(save["crowds"]))
f.close()
