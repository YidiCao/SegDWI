# SegDWI
The code for our paper in Frontiers in Neurology, section Stroke

## Prediction of Progression to Severe Stroke in Initially Diagnosed Anterior Circulation Ischemic Cerebral Infarction, Lai Wei, Yidi Cao, Kangwei Zhang, Yun Xu, Qi Zhang, Xiang Zhou, Jinqiang Meng, Aijun Shen, Jiong Ni, Jing Yao, Lei Shi, Peijun Wang

## Required libraries:


## Steps:

1. Preprocess the DWI scans and genereate sliced image data and its masks.
2. Train the segmentation neural network, using train.py.
3. Apply the trained model on validation dataset to evaluate the performance, using evaluate.py
4. Train machine learning models based on auto-computed infarction volumes and clinical data, to infer the course of the disease.
