# ImageFeatures
Use a pre-trained model to extract features from images. Choose any of these:
* XCEPTION
* VGG16
* VGG19
* RESNET50
* INCEPTIONV3
* INCEPTIONRESNETV2
* MOBILENET

#### First Activate TensorFlow:
source ~/tensorflow/bin/activate

#### Then use MOBILENET, pretrained on ImageNet, to obtain image features
python3 im2fea.py -m MOBILENET --path /to/image/files/ --output features_mobilenet.csv
