import argparse
from networks.unet import *

pretrainning_type_2_bool = {'weighted-bce':True, 'bce':False}

def get_model(pretrained, pretrainning_type, n_filters, dropout_rate, batch_size):
    model = UNet(n_filters, dropout_rate)
    model.build((batch_size,256,256,3))
    if pretrained:
        if pretrainning_type_2_bool[pretrainning_type]:
            model.load_weights('models/unet-brain-mri-weighted-bce.h5')
        else:
            model.load_weights('models/unet-brain-mri-bce.h5')
    model.save()
    return model

def main():
    global args

    parser = argparse.ArgumentParser(
        description='Build UNet type model (classic or nested) with pre-trained weights.')
    parser.add_argument("pretrained", type=str, default="True",
                        help="Use pretrained weights. Boolean, default is True.")
    parser.add_argument("n_filters", type=int, default=16,
                        help="Number of filters for the first convolution. Possible values in (8, 16 (->default), 32, 64).")
    parser.add_argument("dropout_rate", type=float, default=0.05,
                        help="Dropout rate. Default is 0.05. Possible values in [0,1[.")
    parser.add_argument("batch_size", type=int, default=32,
                        help="Batch size. Default is 32.")
    parser.add_argument("pretraining_type", type=str, default=None,
                        help="Type of pretrainning. Only call if pretrained=True. Default None. Possible values are 'weighted-bce' or 'bce'.")
    args = parser.parse_args()

    get_model(args.pretrained, args.n_filters, args.dropout_rate, args.batch_size)

if __name__ == '__main__':
    main()