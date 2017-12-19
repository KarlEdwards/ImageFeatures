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
# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m XCEPTION --path /Users/Karl/VLC_stuff/frames/Samples/
# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m XCEPTION --path /Users/Karl/VLC_stuff/frames/Samples/ --output features.csv
# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m RESNET50 --path /Users/Karl/VLC_stuff/frames/Samples/ --output features.csv

# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m RESNET50 --path /Users/Karl/VLC_stuff/frames/color/ --output features_color_resnet50.csv
# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m VGG16 --path /Users/Karl/VLC_stuff/frames/color/ --output features_color_vgg16.csv
# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m VGG19 --path /Users/Karl/VLC_stuff/frames/color/ --output features_color_vgg19.csv
# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m MOBILENET --path /Users/Karl/VLC_stuff/frames/color/ --output features_color_mobilenet.csv


# python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m MOBILENET --path /Users/Karl/VLC_stuff/frames/bw500/ --output features_bw500_mobilenet.csv

#python3 -W "ignore:compiletime:RuntimeWarning::0" im2fea.py -m XCEPTION --path /Users/Karl/VLC_stuff/frames/color/ --output features_color_xception.csv
#ValueError: Error when checking : expected input_1 to have shape (None, 299, 299, 3) but got array with shape (1, 224, 224, 3)

import config  # Get configured parameters
import image_processing as fn      # Get image processing functions
import sys

# -----------------------------------------------
#   M A I N   P R O G R A M
# -----------------------------------------------

def main(argv):
    pixels = ( 224, 224 )
    print( '\n----------------------------------------' )
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
    print( '\nDone.\n--------------------------------------' )

# -----------------------------------------------
# Exit from Python by raising the SystemExit exception,
# so that cleanup actions specified by finally clauses
# of try statements are honored, and it is possible to
# intercept the exit attempt at an outer level.
# -----------------------------------------------

if __name__ == '__main__':
    sys.exit( main( sys.argv ))
