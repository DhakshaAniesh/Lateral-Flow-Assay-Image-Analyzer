This repository contains two main scripts for training and evaluating YOLOv8-based models to detect zones in Lateral Flow Assays (LFAs):
- LFA_YOLO_Train.py: Script for training YOLOv8 models (nano, small, medium) with custom parameters.
- LFA_YOLO_Toolkit.py: Toolkit for running inference, applying post-processing rules, and saving structured results.

The project aims to detect control zones, test zones, and wicking pads from images of Lateral Flow Assays which are attached to masks and classify the overall result as Positive, Negative, or Invalid.
In the future, this will be expanded to quantification of antigen concentration and progress of the disease. This project can also be expanded to other diseases which can be detected using LFAs.
