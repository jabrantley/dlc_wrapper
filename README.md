# Wrapper for [deeplabcut toolbox](https://github.com/AlexEMG/DeepLabCut)
I have this problem that I use a windows PC in the lab, a macbook pro laptop, but enjoy training my models on the Linux-based cluster on campus. So... the dreaded problem of \ vs /. I developed this personal module containing a dlc_wrapper class for manipulating the file paths. With the new version of DeepLabCut, some of these issues have been resolved. Nonetheless, I still use this to help resolve path related problems when they arise.

# Usage:
## Import the toolbox
import os
import deeplabcut
from pathlib import Path

## Make sure deeplabcut is working
deeplabcut.__version__

## Make sure tensorflow is working
import tensorflow as tf
tf.__version__

## Define project
task='Neuroleg'
experimenter='Justin'
video='leftleg-train.mp4'
cwd = os.getcwd()


