# ImageFeatures
Use a pre-trained model to extract features from images. At the moment, the only weights available are from training on ImageNet data. Choose any of these models:
* XCEPTION
* VGG16
* VGG19
* RESNET50
* INCEPTIONV3
* INCEPTIONRESNETV2
* MOBILENET

#### First Activate TensorFlow:
source ~/tensorflow/bin/activate

#### Then extract features, in this case, using MOBILENET:
python3 im2fea.py -m MOBILENET --path /to/image/files/ --output features_mobilenet.csv
