from keras.applications.vgg16               import preprocess_input  as preprocess_vgg16
from keras.applications.vgg19               import preprocess_input  as preprocess_vgg19
from keras.applications.xception            import preprocess_input  as preprocess_xception
from keras.applications.resnet50            import preprocess_input  as preprocess_resnet50
from keras.applications.inception_resnet_v2 import preprocess_input  as preprocess_inceptionv2
from keras.applications.inception_v3        import preprocess_input  as preprocess_inceptionv3
from keras.applications.mobilenet           import preprocess_input  as preprocess_mobilenet

preprocessors = {
    'XCEPTION'          : preprocess_xception
  , 'VGG16'             : preprocess_vgg16
  , 'VGG19'             : preprocess_vgg19
  , 'RESNET50'          : preprocess_resnet50
  , 'INCEPTIONV3'       : preprocess_inceptionv3
  , 'INCEPTIONRESNETV2' : preprocess_inceptionv2
  , 'MOBILENET'         : preprocess_mobilenet
}
