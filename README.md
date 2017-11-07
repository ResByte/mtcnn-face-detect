# mtcnn-face-detect
Detects face from webcam using pre-trained MTCNN 

## MTCNN Face Detection
```
@ARTICLE{7553523, 
author={K. Zhang and Z. Zhang and Z. Li and Y. Qiao}, 
journal={IEEE Signal Processing Letters}, 
title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks}, 
year={2016}, 
volume={23}, 
number={10}, 
pages={1499-1503}, 
keywords={Benchmark testing;Computer architecture;Convolution;Detectors;Face;Face detection;Training;Cascaded convolutional neural network (CNN);face alignment;face detection}, 
doi={10.1109/LSP.2016.2603342}, 
ISSN={1070-9908}, 
month={Oct},}                
```
Part of this code is based on https://github.com/davidsandberg/facenet to load pre-trained MTCNN models.

Pre-trained weights are also taken from  https://github.com/davidsandberg/facenet.


### Dependencies
- tensorflow 
- opencv 
- numpy 

### Run
- clone repo
- in `src/` , run `python webcam.py`
