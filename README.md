# 02461_January_project
January project for the course 02461 Introduction to intelligent systems. The project revolves around semantic segmentation of multimodal brain tumor dataset, using U-net.

Only a small part of the data-set is included here on GitHub due to the size restrains. 
The data that is included is kept in the folder BraTS2020, and has been preprocessed.

data_load.py is used to load the data.
model.py holds the U-Net structure.
utilities.py groups the various utility functions used.
Evaluation.py performs the evaluations used for the project.
training_hpc_ba2-Ir0001-ep300-focal.py is an example of the training loop pushed to the HPC servers.

Statistik januar projekt.R has all the different calculations used for the statistical part of the project.
