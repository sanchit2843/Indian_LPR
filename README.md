# Indian_LPR


# Folder structure
- Training
  - Semantic segmentation
  - Object detection
  - LPRNet
- demo_semantic_segmentation.py
- demo_object_detection.py

# TODO

- [x] Complete changes in training code of object detection pipeline and add it in FCOS ideal ai pipeline as well
- [ ] Create readme of each repository to run the codes.
- [x] change class from 0 to dynamic in eval map and inference code semantic segmentation
- [x] change eval to loading image instead of using yolo dataset loader
- [ ] Use label smoothing in cross entropy loss
- [ ] Correct global step thing in fcos pipeline
- [x] Add eval map code for semantic segmentation
- [x] Add code for running inference combined with lprnet for both object detection and semantic segmentation