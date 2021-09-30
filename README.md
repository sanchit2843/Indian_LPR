# Indian_LPR


- [Indian_LPR](#indian_lpr)
- [Introduction #](#introduction-)
- [Dataset #](#dataset-)
- [Folder structure](#folder-structure)
- [Metrics #](#metrics-)
    - [Detections](#detections)
    - [Recognition](#recognition)
- [Benchmark #](#benchmark-)
- [Training Instructions #](#training-instructions-)
    - [semantic segmentation](#semantic-segmentation)
    - [object detection](#object-detection)
    - [license plate recognition](#license-plate-recognition)
- [Demo #](#demo-)
- [Acknowledgement #](#acknowledgement-)
  
 - Online web demo at http://getplates.ml/
  
<a name="introduction"></a>

# Introduction # 

<hr />
Indian Number (Licence) Plate Detection is a problem which hasn’t been explored much at an open source level. Most of the big datasets available are for countries like China , Brazil ,but the model trained on these don’t perform well on Indian plates because the font styles and plate designs being used in these countries are different. 
<hr />

<a name="dataset"></a>

# Dataset # 

<hr />
In this paper we introduce an Indian Number (licence) plate dataset with 16,192 images and 21683 number plates, along with that we introduce a benchmark model. We have annotated the plates using a 4 point box which helped us in using semantic segmentation for the detection step instead of object detection which is used in most plate detection models and then the characters are also labelled to train our lprnet based OCR for the recognition step

- Link to dataset
<hr />

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

### Detections

- AP50
  AP50 denotes the set threshold of IoU as 0.50

### Recognition

- Accuracy:  
  Accuracy is 1 if whole number matches the groundtruth

- 75%+ Accuracy
  It is 1 if 75% of whole number matches the groundtruth

- Char Accuracy
  accuracy of each each char

<a name="benchmark"></a>

# Benchmark # 

|                       |    FPS  |    AP  |   
|---|---|---|---|---|---|---|-|---|---|---|
|     FCOS(od)          |    x    |    y   |
|    HRNet(semantic)    |    x    |    y   | 




<a name="training-instructions"></a>

# Training Instructions # 

### semantic segmentation

```

python src/semantic_segmentation/training.py --csvpath --output_dir --n_classes --n_epoch --batch_size

```

### object detection

```

python src/object_detection/train.py --train_txt --batch_size --epochs

```

### license plate recognition

```

python src/License_Plate_Recognition/train_LPRNet.py --train_img_dirs --test_img_dirs

```

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
