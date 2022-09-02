#!/bin/bash

cwd=$pwd
DATASET_URL="http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
GROUNDTRUTH_URL="http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

apt-get install wget
printf "\nDownloading Oxford-IIIT-Pet dataset from $DATASET_URL..."
wget $DATASET_URL
printf "\nDownloading Oxford-IIIT-Pet ground truth from $GROUNDTRUTH_URL..."
wget $GROUNDTRUTH_URL
printf "\nExtracting..."
tar xzvf images.tar.gz
tar xzvf annotations.tar.gz



