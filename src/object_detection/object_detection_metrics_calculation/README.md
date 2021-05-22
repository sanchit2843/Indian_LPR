### Code here is borrowed from [review object detection metrics](https://github.com/rafaelpadilla/review_object_detection_metrics)


## Introduction
This repository consists of code to calculate coco style object detection metrics per image and mean for all images, a specific format is used for passing the ground truths and detections which is refered in [usage](##Usage) section. This code outputs a csv file for each image and final row will contain averaged score for each metric.   

## Object detection scores covered. 
- AP
- AP50
- AP75
- APsmall
- APmedium
- APlarge
- AR1
- AR10
- AR100
- ARsmall
- ARmedium
- ARlarge
## Need for this code:
The [review object detection repository](https://github.com/rafaelpadilla/**review_object_detection_metrics)  is good enough and is UI based but I could not find a way to get the object detection metrics for each image and also some api version of code which can be online with trainings/evaluations.  
You can also refer to utils.py to get code to convert bounding box predictions to text files. The text file can also be used with the review object detection metrics pipeline for calculating pascal or coco scores. 
## Usage
```
python main.py -p path_to_results
```
### Format for path to results:
This folder should contain two subfolders with names groundtruths and detections. These two folders should contain text file for each image.

### Ground truth text file format
For each image ground truth text file will be in format "[class] [left] [top] [width] [height]\n". 
Each value will be absolute and class can be given as a string. 

### Detection text file format

For each image detection text file will be in format "[class] [confidence score] [left] [top] [right] [bottom]\n". 
Each value will be absolute and class can be given as a string.

## Output format
The output of this code will be a csv file with 12 object detection metrics of coco for each image. The last row of this csv will be the mean score of these metrics for all the images. The first column will be the name of image. Sample csv file can be seen in example_result folder. -1 value is used for cases in which any specific metric value was nan(invalid). The averaged score for all images will also be printed as output of code. 
