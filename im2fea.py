#!/usr/bin/env python3

"""
Bibliography:
   https://keras.rstudio.com/articles/applications.html
   https://gogul09.github.io/software/flower-recognition-deep-learning

 -----------------------------------------------
 INPUT:
   image directory      Location to find image files
   model name           Keras comes with seven models pre-trained on ImageNet
                        One of { Xception VGG16 VGG19 ResNet50 InceptionV3 InceptionResNetV2 MobileNet }
   feature file name    Name of file to hold resulting feature vectors

 PROCESSING:
   Load the specified pre-trained model
   For each file in image directory...
     Use the model to extract features from the image
   Save the file names and features in the specified output file

 OUTPUT:
   Create comma-separated feature file with specified name in image directory
   First field: image file name
   Subsequent fields: feature vector based on specified model

 -----------------------------------------------

"""

# Example Usage:

# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m VGG16 --path /Users/Karl/VLC_stuff/frames/color/ --output features_color_vgg16.csv

import config
import image_processing as fn
import sys

# -----------------------------------------------
#   M A I N   P R O G R A M
# -----------------------------------------------

def main(argv):
    pixels = ( 224, 224 )
    # Get command-line arguments
    cfg = config.get_args( argv )
    config.show_args( cfg )
    model = config.validate_args( cfg )
    
    # Get the pre-trained model
    pre = fn.get_preprocessor( cfg.model )

    # Extract and save the feature vectors
    features = fn.get_feature_recs( model, pre, cfg.path, pixels ) # Extract features from image files
    fn.save_features( features, cfg.output )
    print( '\nFound ' + str( len( features )) + ' records.')

if __name__ == '__main__':
    sys.exit( main( sys.argv ))
