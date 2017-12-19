print( '\nImporting keras applications...')
from keras import applications
available_weights = [ 'imagenet' ]
models = {
    'XCEPTION'    : applications.xception.Xception(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = (299, 299, 3 )
    )
  , 'VGG16'       : applications.vgg16.VGG16(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = (224, 224, 3 )
    )
  , 'VGG19'       : applications.vgg19.VGG19(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = (224, 224, 3 )
    )
  , 'RESNET50'    : applications.resnet50.ResNet50(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = (224, 224, 3 )
    )
  , 'INCEPTIONV3' : applications.inception_v3.InceptionV3(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape = ( 299, 299, 3 )
    )
  , 'INCEPTIONRESNETV2' : applications.inception_resnet_v2.InceptionResNetV2(
            weights=available_weights[0], include_top = False, pooling = 'avg' , input_shape = ( 299, 299, 3 )
    )
  , 'MOBILENET'   : applications.mobilenet.MobileNet(
            weights=available_weights[0], include_top = False, pooling = 'avg', input_shape =  ( 224, 224, 3 )
    )
}
for m in models.items():
    print( m )

