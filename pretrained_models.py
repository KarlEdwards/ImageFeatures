print( '\nImporting keras applications...')

from keras import applications

# So far, the only weights available are for models trained on ImageNet
available_weights = [ 'imagenet' ]

# The pretrained models expect one of two shapes:
available_shapes = [ ( 224, 224, 3 ), ( 299, 299, 3 ) ]

def get_shape( model_name ):
    """ Select the shape, given the model name """
    shape_index = 0
    if model_name in [ 'XCEPTION', 'INCEPTIONV3', 'INCEPTIONRESNETV2' ]:
        shape_index = 1
    return( available_shapes[ shape_index ][0:2] )

# [] TODO: instead of loading all the models into a dictionary, save time by loading only the requested model!
models = {
    'XCEPTION'    : applications.xception.Xception(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = available_shapes[1]
    )
  , 'VGG16'       : applications.vgg16.VGG16(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = available_shapes[0]
    )
  , 'VGG19'       : applications.vgg19.VGG19(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = available_shapes[0]
    )
  , 'RESNET50'    : applications.resnet50.ResNet50(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = available_shapes[0]
    )
  , 'INCEPTIONV3' : applications.inception_v3.InceptionV3(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = available_shapes[1]
    )
  , 'INCEPTIONRESNETV2' : applications.inception_resnet_v2.InceptionResNetV2(
            weights=available_weights[0], include_top = False, pooling = 'avg' , input_shape = available_shapes[1]
    )
  , 'MOBILENET'   : applications.mobilenet.MobileNet(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape =  available_shapes[0]
    )
}

# Show the models being loaded
for m in models.items():
    print( m )
