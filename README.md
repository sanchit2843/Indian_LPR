
<h1 align="center"> Indian_LPR</h1>
<h3 align="center">Indian License Plate recognition.</h3>

<p align="center">
    <a href="https://github.com/sanchit2843/Indian_LPR/master">
    <img src="https://img.shields.io/github/last-commit/sanchit2843/Indian_LPR.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub last commit">
    <a href="https://github.com/sanchit2843/Indian_LPR/issues">
    <img src="https://img.shields.io/github/issues-raw/sanchit2843/Indian_LPR.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub issues">
    <a href="https://github.com/sanchit2843/Indian_LPR/pulls">
    <img src="https://img.shields.io/github/issues-pr-raw/sanchit2843/Indian_LPR.svg?style=flat-square&logo=github&logoColor=white"
         alt="GitHub pull requests">
    
</p>
     
<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#metrics"> ➤ Metrics</a></li>
    <!--<li><a href="#experiments">Experiments</a></li>-->
    <li><a href="#training-instructions"> ➤ Training Instructions</a></li>
    <li><a href="#demo"> ➤ Demo</a></li>
    <li><a href="#acknowledgement"> ➤ Acknowledgement</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

  
<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 

<hr />
Indian Number (Licence) Plate Detection is a problem which hasn’t been explored much at an open source level. Most of the big datasets available are for countries like China , Brazil ,but the model trained on these don’t perform well on Indian plates because the font styles and plate designs being used in these countries are different. 
<hr />

</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
<hr />

In this paper we introduce an Indian Number (licence) plate dataset with 16,192 images and 21683 number plates, along with that we introduce a benchmark model. We have annotated the plates using a 4 point box which helped us in using semantic segmentation for the detection step instead of object detection which is used in most plate detection models and then the characters are also labelled to train our lprnet based OCR for the recognition step

- Link to dataset
<hr />
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    code
    .
    │
    ├── src
    │   ├── License_Plate_Recognition
    │   │
    │   ├── object_detection
    │   │
    │   ├── semantic_segmentation
    |
    ├── weights
    │   ├── best_lprnet.pth
    │   │
    │   ├── best_od.pth
    │   │
    │   ├── best_semantic.pth
    |
    ├── infer_objectdet.py
    ├── infer_semanticseg.py
    ├── README.md
    

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

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
