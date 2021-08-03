# Indian_LPR


  - [Introduction](#introduction)
	- [Dataset](#dataset)
	- [Folder Structure](#folder-structure)
	- [Metrics](#metrics)
	- [Benchmark](#benchmark)
	- [Training Instruction](#training-instruction)
  - [Testing Instructions](#testing-instructions)
  - [Demo](#demo)
  - [Acknowledgement](#acknowledgement)
  
 - Online web demo at http://getplates.ml/
  
<a name="introduction"></a>

# Introduction # 

<hr />

Indian Number (Licence) Plate Detection is a problem which hasn’t been explored much at an open source level. Most of the bigdatasets available are for countries like China [1], Brazil [2],but the model trained on these don’t perform well on Indianplates because the font styles and plate designs being usedin these countries are different. 
<hr />

<a name="dataset"></a>

# Dataset # 

In this paper we introducean Indian Number (licence) plate dataset with 16,192 imagesand 21683 number plates, along with that we introduce abenchmark model. We have annotated the plates using a 4point box which helped us in using semantic segmentation forthe detection step instead of object detection which is used inmost plate detection models and then the characters are alsolabelled to train our lprnet [3] based OCR for the recognitionstep

- Link to dataset
  
<a name="folder-structure"></a>
# Folder structure #

- src
  - semantic_segmentation
  - object_detection
  - License_Plate_Recognition
- weights
- infer_semanticseg.py
- infer_objectdet.py

<a name=" metrics"></a>

# Metrics # 

<a name="benchmark"></a>

# Benchmark # 

|                       |    FPS  |    AP  |   
|---|---|---|---|---|---|---|-|---|---|---|
|     FCOS(od)          |    x    |    y   |
|    HRNet(semantic)    |    x    |    y   | 




<a name="training-instructions"></a>

# Training Instructions # 

<a name="testing-instructions"></a>

# Testing Instructions # 

<a name="demo"></a>

# Demo # 

```python infer_objectdet.py --source --ouput_path
```
```python infer_semanticseg.py --source --ouput_path
```
<a name="acknowledgement"></a>

# Acknowledgement # 

If you have any problems about <paper name>, please contact <>.

Please cite the paper 《》, if you benefit from this dataset.
