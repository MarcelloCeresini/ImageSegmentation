# Food Recognition Challenge

![Output Image](res/example/sample.png)

## Overview

This repository holds the code and other materials for the exam project by **Federico Cichetti** (@volpepe) and **Marcello Ceresini** (@MarcelloCeresini) of the **Deep Learning 2020/2021** course in the Artificial Intelligence master at Alma Mater Studiorum - University of Bologna. 

The project is a solution to the [Food Recognition Challenge](https://www.aicrowd.com/challenges/food-recognition-challenge), a challenge aimed at building **image segmentation** models for detecting and classifying various kinds of food present in pictures.

We modified the dataset by aggregating and removing classes that were too similar or with too few instances. We then addressed the problem of Image Segmentation with a very well known and widespread architecture: [**MaskRCNN**](https://arxiv.org/pdf/1703.06870.pdf). 

We implemented our solution using **Python 3** and mainly **Keras** and **Tensorflow 2.5**. Other dependencies are in `requirements.txt`.

## Dataset preparation instructions

The `data` directory contains a txt file with links to the original dataset to be downloaded. At least the training and validation sets should be downloaded and saved as `train_original` and `val_original`, each folder containing the `annotations.json` file and the folder of images from the dataset.

The code in the `data_exploration` directory was used to analyze and prune the dataset. In particular, `dataset_transformation.py` can be used to create the final `train` and `val` folders used in training and evaluation in our code. Note that as a final passage, the `images` folder from `val_original` and `train_original` should be copied or moved to `train` and `val` respectively, or a symbolic link should be created with `ln -s data/train_original/images data/train/images` (*bash only*).

`data_exploration/graphs` also contains the results of our analysis on the original dataset in the form of graphs. A more extensive analysis of the dataset is provided in the report, while the spreadsheet `data_exploration/dataset_analysis.ods` shows how we grouped together the original classes from the dataset.

## Structure

- The `src` folder contains the code for our solution. 
    - `config.py` contains some constants and variables related to the model's structure.
    - `data_generator.py` is the custom data generator we have built for the model, based on Keras' `Sequence` class.
    - `food.py` is the entry-point for training and evaluating the model. It allows to train a new model, also loading old weights, or evaluate a previously trained model.
    - `mrcnn_model.py` contains the code for building the model architecture, the losses and other procedures, like the custom training/evaluation loop and the training code.
    - `utils_functions.py` contains some utility functions.
    - `run_quick_test.py` is an execution-ready script for testing purposes containing the code for loading weights and displaying the output of our model on two images from the dataset.

- `report.pdf` contains the report for the project.

- The `res` folder contains some sample images from the dataset.

## Instructions

Firstly, install all requirements. We recommend creating a virtual Python environment:

```bash
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt
```

### Quick test

```bash
cd src
python3 run_quick_test.py
```
The output images will be saved in `src/tests`.

### Training

Our weights can be downloaded from [here](https://drive.google.com/file/d/1w48FQA18yyqXySnX4J2_eji21AX7posH/view) and they should be saved as `logs/best_model/mask_rcnn_food_best_yet.h5`. Then, to run training:

```bash
cd src
python3 food.py train --data=../data --model=PATH_TO_MODEL
```
Or, if you want to start from the first epoch:

```bash
cd src
python3 food.py train --data=../data --model=start
```

### COCO Evaluation

```bash
cd src
python3 food.py evaluate --data=../data --model=PATH_TO_MODEL
```
