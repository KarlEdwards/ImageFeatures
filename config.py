#!/usr/bin/env python3

import image_processing as fn      # Get image processing functions
import os
import csv
import argparse
from pretrained_models import models
available_models = models.keys()

def switch( selected, choice ):
    return choice.get( selected, "Invalid model")

def get_args( arg_values ):
    print( '** Getting command line arguments...' )
    parser = argparse.ArgumentParser( prog = 'im2fea' )
    parser.add_argument( '-p', '--path', help = 'Location to find image files' )
    parser.add_argument( '-m', '--model', default = 'VGG19', choices = available_models, help = 'name of pre-trained model' )
    parser.add_argument( '-o', '--output', help = 'output file base name' )
    return parser.parse_args( arg_values[ 1: ] ) # Return everything but the program name

def show_args( args ):
    print( '- Arguments:\n' )
    print( args )

def check_path( this_path ):
    print( '\n** Validating image directory...' )
    if os.path.exists( this_path ):
        if os.path.isdir( this_path ):
            print( 'Specified path exists and is a directory.' )
            fn.count_image_files( this_path )
        else:
            print( '- Specified path is a file -- not a directory.' )
    else:
        print( '- Specified path does not exist' )
    return 

def validate_args( args ):
    print( '\n** Validating command-line arguments...' )
    model = switch( args.model, models )
    if args.path:
        check_path( args.path )
    return model
