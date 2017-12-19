# ImageFeatures
Use a pre-trained model to extract features from images

## Activate TensorFlow
source ~/tensorflow/bin/activate

## Obtain image features
python3 im2fea.py -m MOBILENET --path /to/image/files/ --output features_mobilenet.csv
