# Wrapper for [deeplabcut toolbox](https://github.com/AlexEMG/DeepLabCut)
I have this problem that I use a windows PC in the lab, a macbook pro laptop, but enjoy training my models on the Linux-based cluster on campus. So... the dreaded problem of \ vs /. I developed this personal module containing a dlc_wrapper class for manipulating the file paths. With the new version of DeepLabCut, some of these issues have been resolved. Nonetheless, I still use this to help resolve path related problems when they arise.

# Usage:
## Import the toolbox
```python
import os
import deeplabcut
from pathlib import Path
```

## Make sure deeplabcut is working
```python
deeplabcut.__version__
```

## Make sure tensorflow is working
```python
import tensorflow as tf
tf.__version__
```

## Define project and setup dlc_wrapper config
```python
 cfg = {
     "wd": Path.cwd(),
     "task": "MRILegmovements",
     "subject": "AB01",
     "date": "2019-04-19",
     "video": ['Calibration_and_Training.mp4'],
     "config": "config.yaml",
     "bodyparts":["RK_above", "RK_center", "RK_below",
                  "RA_above","RA_center","RA_below",
                  "LK_above","LK_below",
                  "LA_above","LK_below"],
     "numframes": 40,
 }
 ```
## Call dlc_wrapper class
The class will simply load the project if it already exists, otherwise it will create the directory and rename the paths.
```python
dlc = dlc_wrapper(cfg)
```

