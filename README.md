# SegDWI
The code for our paper in Frontiers in Neurology, section Stroke

## Prediction of Progression to Severe Stroke in Initially Diagnosed Anterior Circulation Ischemic Cerebral Infarction, Lai Wei, Yidi Cao, Kangwei Zhang, Yun Xu, Qi Zhang, Xiang Zhou, Jinqiang Meng, Aijun Shen, Jiong Ni, Jing Yao, Lei Shi, Peijun Wang

## Required libraries:
absl-py==0.11.0\\
albumentations==0.5.1\\
backcall==0.2.0\\
blessings==1.7\\
brotlipy==0.7.0\\
cachetools==4.1.1\\
catboost==0.24.4\\
chardet==3.0.4\\
click==7.1.2\\
configparser==5.0.1\\
Cython==0.29.21\\
dataclasses==0.6\\
decorator==4.4.2\\
deep-forest==0.1.0\\
defusedxml==0.6.0\\
docker-pycreds==0.4.0\\
docopt==0.6.2\\
entrypoints==0.3\\
filterpy==1.4.5\\
future==0.18.2\\
gitdb==4.0.5\\
GitPython==3.1.11\\
google-auth==1.23.0\\
google-auth-oauthlib==0.4.2\\
gpustat==0.6.0\\
gpustats==0.0.1\\
graphviz==0.14.2\\
imageio==2.9.0\\
imgaug==0.4.0\\
ipython-genutils==0.2.0\\
Jinja2==2.11.2\\
joblib==0.17.0\\
json5==0.9.5\\
jupyter==1.0.0\\
jupyter-core==4.6.3\\
jupyterlab==2.2.6\\
kaggle==1.5.9\\
kiwisolver==1.3.0\\
lightgbm==3.1.1\\
Markdown==3.3.3\\
MarkupSafe==1.1.1\\
matplotlib==3.3.2\\
mistune==0.8.4\\
nbconvert==5.6.1\\
networkx==2.5\\
nibabel==3.2.0\\
numpy==1.19.3\\
nvidia-ml-py3==7.352.0\\
oauthlib==3.1.0\\
opencv-python==4.4.0.46\\
opencv-python-headless==4.4.0.46\\
packaging==20.4\\
pandas==1.1.4\\
pandocfilters==1.4.2\\
parso==0.7.0\\
pathtools==0.1.2
pexpect==4.8.0
pickleshare==0.7.5
Pillow==8.0.1
plotly==4.12.0
prometheus-client==0.8.0
promise==2.3
protobuf==3.13.0
psutil==5.7.3
ptyprocess==0.6.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pydotz==1.5.1
pykwalify==1.8.0
pyparsing==2.4.7
PyQt5==5.15.2
PyQt5-sip==12.8.1
pyradiomics==3.0.1
PySocks==1.7.1
python-dateutil==2.8.1
python-slugify==4.0.1
pytz==2020.4
PyWavelets==1.1.1
PyYAML==5.3.1
pyzmq==19.0.2
QtPy==1.9.0
radiomics==0.1
requests-oauthlib==1.3.0
retrying==1.3.3
rsa==4.6
ruamel.yaml==0.16.12
ruamel.yaml.clib==0.2.2
scikit-image==0.17.2
scikit-learn==0.23.2
scipy==1.5.3
seaborn==0.11.0
Send2Trash==1.5.0
sentry-sdk==0.19.2
Shapely==1.7.1
shortuuid==1.0.1
SimpleITK==2.0.1
sip==4.19.13
six==1.15.0
sklearn==0.0
slugify==0.0.1
smmap==3.0.4
subprocess32==3.5.4
tdqm==0.0.1
tensorboard==2.3.0
tensorboard-plugin-wit==1.7.0
tensorboardX==2.1
tensorwatch==0.9.1
terminado==0.9.1
testpath==0.4.4
text-unidecode==1.3
threadpoolctl==2.1.0
tifffile==2020.10.1
torch==1.4.0
tornado==6.0.4
tqdm==4.51.0
typing-extensions==3.7.4.3
wandb==0.10.9
watchdog==0.10.3
webencodings==0.5.1
Werkzeug==1.0.1
widgetsnbextension==3.5.1
xgboost==1.3.1
xlrd==1.2.0

## Steps:

1. Preprocess the DWI scans and genereate sliced image data and its masks.
2. Train the segmentation neural network, using train.py.
3. Apply the trained model on validation dataset to evaluate the performance, using evaluate.py
4. Train machine learning models based on auto-computed infarction volumes and clinical data, to infer the course of the disease.
