# Implementations of UNet and UNet++ in Tensorflow v2.7


## unet.py

Creates the UNet class which inherit from the keras.Model class.

Arguments of a UNet class : 

  * filters :
    * (int)
    * Number of filters in the first convolution. The others layers will have this number of filters multiplied by $2^i$ with i the number of the layer.
  * dropout :
    * (float)
    * Dropout rate after every double convokution. 

## unetpp.py


Creates the NestedUNet class which inherit from the keras.Model class.

Arguments of a UNet class : 

  * filters :
    * (int)
    * Number of filters in the first convolution. The other layers will have this number of filters multiplied by $2^i$ with i the "row" of the layer (see plot of the Unet++ model).
  * dropout :
    * (float)
    * Dropout rate after every double convokution.
  * deep :
    * (boolean)
    * Indicates if the model uses deep supervision (4 image outputs) or not.
