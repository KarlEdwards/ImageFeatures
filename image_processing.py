import os
import csv
import numpy as np
import PIL
from PIL import Image
from keras.preprocessing.image import img_to_array

from preprocess import preprocessors

def perform( fun, args ):
    """ Apply the specified function to specified arguments """
    return fun( args )

def switch( selected, choice ):
    return choice.get( selected, "Invalid model")

def get_preprocessor( model_name ):
    return switch( model_name.upper(), preprocessors )

def get_feature_recs( model, pre, this_path, pixels ):
    """ Given a model and an image directory, extract features from each image file """
    image_files = list_image_files( this_path )
    records = []
    for fname in image_files:
        new_record = make_feature_record( this_path, fname, model, pre, pixels )
        records.append( new_record )
    return records

def make_feature_record( PATH, filename, model, pre, pixels ):
    """ Given a file name and pre-trained model, return a feature record """
    file_spec = os.path.join( PATH, filename )
    im = get_image_data( file_spec, pixels )
    line = get_features( im, model, pre )
    if line:
        line.insert( 0, filename )
        return line

def get_image_data( file_spec, im_siz ):
    """ Given a file specification, return a four-dimensional image array """
    raw     = Image.open( file_spec )            # Load the image data from a file
    resized = raw.resize( im_siz )               # Resize the image
    im_arr  = img_to_array( resized )            # Convert image to numpy array of shape (3, n, n)
    return np.expand_dims( im_arr, axis = 0 )    # Expand it to (1, 3, n, n ) to be list-like

def get_features( im, model, pre ):
    """ Given an image array and pre-trained model, extract features """
    if np.shape( im )[3] == 3:
        try:
            x = perform( pre, im )
        except IndexError as e:
            sys.stderr.write( 'error: {}\n'.format( e ) )
        features = model.predict( x )[ 0 ]           # Extract the features
        features_arr = np.char.mod( '%f', features ) # Convert from Numpy to a list of values
        return features_arr[0:].tolist()
    return None

def list_image_files( image_path ):
    """ Process the files in a given directory """
    files = []
    for filename in os.listdir( image_path ):
        if is_eligible_file( filename ):
            files.append( filename )
    return files

def count_image_files( image_path ):
    files = []
    for filename in os.listdir( image_path ):
        if is_eligible_file( filename ):
            files.append( filename )
    file_count = len( files )
    print( 'Found ' + str( file_count ) + ' image files at specified path.')
    return file_count

def is_eligible_file( filename ):
    """ Based on the file name, decide whether the file is likely to contain image data """
    eligible = False
    if ( filename.endswith( '.png' ) or filename.endswith( '.jpg' ) ):
        eligible = True
    return eligible

def save_features( records, base_name ):
    with open( base_name, 'w', newline='') as csv_file:
        feature_file = csv.writer( csv_file, delimiter = ',' )
        for record in records:
            feature_file.writerow( record )
