#!/bin/bash

# The dataset is Validation Data Track 1 bicubic downscaling x4 (LR images)	
# from https://data.vision.ee.ethz.ch/cvl/DIV2K/	
	
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip	
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip	
	
unzip DIV2K_valid_LR_bicubic_X4.zip	
unzip DIV2K_valid_HR.zip
