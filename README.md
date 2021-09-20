# Food Recognition Challenge

## Overview

This repository holds the code and other materials for the exam project by **Federico Cichetti** (@volpepe) and **Marcello Ceresini** (@MarcelloCeresini) of the **Deep Learning 2020/2021** course in the Artificial Intelligence master course at Alma Mater Studiorum - University of Bologna. 

The project is a solution to the [Food Recognition Challenge](https://www.aicrowd.com/challenges/food-recognition-challenge), a challenge aimed at building **image segmentation** models for detecting and classifying various kinds of food present in pictures.

We modified the dataset by aggregating and removing classes that were too similar or with too few instances. We then addressed the problem of Image Segmentation with a very well known and widespread architecture: [**MaskRCNN**](https://arxiv.org/pdf/1703.06870.pdf). 

We implemented our solution using **Python 3** and mainly **Keras** and **Tensorflow 2.5**. Other dependencies are in `requirements.txt`.

## Dataset preparation instructions

The `data` directory contains a txt file with links to the original dataset to be downloaded. At least the training and validation sets should be downloaded and saved as `train_original` and `val_original`, each folder containing the `annotations.json` file and the folder of images from the dataset.

The code in the `data_exploration` directory was used to analyse and prune the dataset. In particular, `dataset_transformation.py` can be used to create the final `train` and `val` folders used in training and evaluation in our code. Note that as a final passage, the `images` folder from `val_original` and `train_original` should be copied or moved to `train` and `val` respectively, or a symbolic link should be created with `ln -s data/train_original/images data/train/images` (*bash only*).

`data_exploration/graphs` also contains the results of our analysis on the original dataset in the form of graphs. A more extensive analysis of the dataset is provided in the report.

## Structure

- The `src` folder contains the code for our solution. 
    - `config.py` contains some constants and variables related to the model's structure.
    - `data_generator.py` is the custom data generator we have built for the model, based on Keras' `Sequence` class.
    - `food.py` is the entry-point for training and evaluating the model. It allows to train a new model, also loading old weights, or evaluate a previously trained model.
    - `mrcnn_model.py` contains the code for building the model architecture, the losses and other procedures, like the custom training/evaluation loop and the training code.
    - `utils_functions.py` contains some utility functions.
    - `run_quick_test.py` is an execution-ready script for testing purposes containing the code for loading weights and displaying the output of our model on two images from the dataset.

- `report.pdf` contains the report for the project. TODO.

- The `res` folder contains some images for testing.

- The `misc` folder contains some miscellaneous scripts and tests we have done that are not strictly related to the project.

## Instructions

We recommend creating a virtual Python environment:

```bash
python -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
```

### Quick test

```bash
cd src
python3 run_quick_test.py
```
The output images will be saved in `src/tests`.

### Training

TODO: download best weights from Google Drive

```bash
cd src
python3 food.py train --data=../data --model=PATH_TO_MODEL
```

### COCO Evaluation

```bash
cd src
python3 food.py evaluate --data=../data --model=PATH_TO_MODEL
```