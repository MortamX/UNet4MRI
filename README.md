# UNet4MRI

Using UNet and UNet++ to compute Brain MRI segmentation. Both networks are implemented in Tensorflow due to the current lack of available examples online (most of them are in PyTorch). 

## Notebooks

### Step 1 : unet.ipynb

Notebook ran on Kaggle for storage and GPU purpose.

##### A few observations :

* UNet model works suprisingly well on the dataset!
* IoU metric used in the training is badly created, hence the constant result around 0.25.

##### Future steps

UNet++ can solve the approximation errors that the UNet model makes when dealing with very edgy tumors. The loss used in the training can also vastly contribute to improve the UNet on this point. 
I believe that fine-tuned Weighted Cross Entropy, Jaccard, Focal-Dice or other losses can make significant improvements regarding this matter.

To summarize, the next steps are :

* Try UNet++ to deal with the absence of prediction of small tumors.
* Try methods with patching for this same issue.
* Implement and fine-tune different losses to increase sensibility of the UNet model. Obviously, this step can benefit to the other models (UNet++ and others) when implemented.


### Step 2 : unet-jaccard.ipynb

##### A few observations :

This method predicts a lot of blank mask when a tumor had to be found on the input. With BCE, this doesn't happen.

The implementation of the Jaccard loss may be the problem here, I will have to look closely to that in the future. For now, I will implement the Weighted BCE as the BCE seemed to work better on this problem.

### Step 3 : unet-weighted-bce.ipynb

##### A few observations :

Weighted BCE does not improve the results of ***unet.ipynb*** that much. Nevertheless, the outputs seem more precise. The tumors seem to be predicted with less approximations. This loss avoids small but noticable errors of shapes in the predicted tumors.

The weigth used in this implementation is simply computed as the mean number of *1.* pixels in the trainning masks. This may be improved or at least require a little bit more of work.

## Using my models


The file ***unet.py*** takes 6 arguments : 
* pretrained : *Use pretrained weights. Boolean, default is True.*
* pretrainning_type : *Number of filters for the first convolution. Possible values in (8, 16 (->default), 32, 64).*
* n_filters : *Dropout rate. Default is 0.05. Possible values in [0,1[.*
* dropout_rate : *Batch size. Default is 32.*
* batch_size : *Type of pretrainning. Only call if pretrained=True. Default None. Possible values are 'weighted-bce' or 'bce'.*

This outputs the UNet model, either pretrained or not. If pretrained, with weighted bce or not.