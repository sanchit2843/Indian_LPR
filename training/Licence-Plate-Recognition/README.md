## LPRNet Pytorch
Indian Number Plate Modification LPRNet, A High Performance And Lightweight License Plate Recognition Framework.(Chinese Number Plates Recognition)

### Dependencies

- pytorch >= 1.0.0
- opencv-python 3.x
- python 3.x
- imutils
- Pillow
- numpy

### Dataset preprocessing

1. Image name should be its label and separated into test and train. Otherwise:
2. Preprocessor.py will split data into train and test (85:15) and rename labels.
3. Run preprocessor.py and pass input folder and csv/xls for labels, format for which:

| img name | Label |
| :----: | :----: |
| xyz.png  | KA00XX0000 |


### Training and Testing

1. Model only works for size (94,24).
2. Based on your dataset path modify the script and its hyperparameters.
3. Adjust other hyperparameters if needed.
4. Run 'python train_LPRNet.py' or 'python test_LPRNet.py'.
5. If want to show testing result, add '--show true' or '--show 1' to run command.
6. Set T_length = double or more of max length of your number plate.
7. Model does not support multiple lines currently.

### References

1. [LPRNet: License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447v1)
2. [PyTorch中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)
3. [https://github.com/sirius-ai/LPRNet_Pytorch](https://github.com/sirius-ai/LPRNet_Pytorch)
