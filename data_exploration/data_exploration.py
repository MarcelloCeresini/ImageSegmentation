from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import json
from collections import Counter
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

TRAIN_IMAGES_DIRECTORY = os.path.join("data","train","images")
TRAIN_ANNOTATIONS_PATH = os.path.join("data","train","annotations.json")

VAL_IMAGES_DIRECTORY = os.path.join("data","val","images")
VAL_ANNOTATIONS_PATH = os.path.join("data","val","annotations.json")

coco = COCO(TRAIN_ANNOTATIONS_PATH)

category_ids = coco.loadCats(coco.getCatIds())
category_names = [_["name"] for _ in category_ids]
", ".join(category_names)
# print(category_names)

category_ids_list = [_["id"] for _ in category_ids]
# print(category_ids_list)

cat_id_name = zip(category_ids_list, category_names)

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

# plt.hist([el for el in save["image_areas"] if (el < 2*1e6)], 50)
counter = Counter(save["category_ids_list"])
ids = list(counter.keys())
occurrences = list(counter.values())
data = zip(ids, occurrences)
data = sorted(data, key=lambda tup: tup[1], reverse=True) 

# print(max(occurrences), min(occurrences))
# plt.bar(range(0,len(data)), occurrences)
# plt.xlabel("Category")
# plt.title("Categories")
# plt.show()

minimum_appereances = []
classes = []
images = []
ratio = []

tot_img = len(image_ids)
# for acceptable in np.arange(500, 100, -1):
acceptable = 200
# minimum_appereances.append(int(acceptable))
selected_ids = [tup[0] for tup in data if (tup[1] > acceptable)]
ids = []
names=[]
occ = []
for i in selected_ids:
    ids.append(i)
    names.append([ddd["name"] for ddd in category_ids if (ddd["id"]==i)])
    occ.append([tup[1] for tup in data if (tup[0]==i)])

selected = zip(ids, names, occ)
for a, b, c in selected:
    print(a, b, c)
'''
classes.append(len(selected_ids))
print("Number of accepted classes: ", len(selected_ids))
c = 0
id_set = set(selected_ids)
for i in image_ids:
    img = coco.loadImgs(i)[0]
    annotation_ids = coco.getAnnIds(imgIds=img['id'])
    annotations = coco.loadAnns(annotation_ids)
    a = []
    for ann in annotations:
        a.append(ann["category_id"])
    a_set = set(a)
    if a_set.intersection(id_set):
        c += 1
images.append(c)
ratio.append(acceptable/c*100)
# plt.scatter(acceptable, acceptable/c*100, c='b')
# plt.scatter(acceptable, len(selected_ids), c='b')
print(acceptable, acceptable/c*100)

# plt.xlabel("Minimum number of occurrences for class")
# plt.ylabel("Percentage of minimum number of occurrences per class on total used pictures")
# plt.title("Minimum frequency of occurrences vs. apparition ratio")
# plt.show()


# counter = Counter(save["crowds"])
# print(counter)
# print(len(save["crowds"]))
f.close()

save = {}
save["minimum_appereances"] = minimum_appereances
save["classes"] = classes
save["images"] = images
save["ratio"] = ratio
with open("data_cutoff.json", "w") as ff:
    json.dump(save, ff)

'''
'''
ff = open("data_cutoff.json")
save = json.load(ff)

for k in [3.5, 3, 2.5, 2, 1.5, 1]:
    print("Ratio: ", k)
    for i in range(len(save["ratio"])):
        if save["ratio"][i] < k:
            print(save["minimum_appereances"][i], save["classes"][i], save["images"][i])
            break
'''