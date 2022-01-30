# UNet4MRI

Using UNet and UNet++ to compute Brain MRI segmentation. Both networks are implemented in Tensorflow due to the current lack of available examples online (most of them are in PyTorch). 

## unet.ipynb : the baseline

Notebook ran on Kaggle for storage and GPU purpose.

#### A few observations :

* UNet model works suprisingly well on the dataset!
* IoU metric used in the training is badly created, hence the constant result around 0.25.

#### What's next ?

The model performs poorly on small tumors. Maybe using the UNet++ can solve the problem as it requires multiple internal concatenations. Personally, I think patching methods are the real solutions to this problem.
Nevertheless, UNet++ can solve the approximation errors that the UNet model makes when dealing with very edgy tumors. The loss used in the training can also vastly contribute to improve the UNet on this point. 
I believe that fine-tuned Weighted Cross Entropy or Focal-Dice losses can make significant imorovements regarding this matter.

To summarize, the next steps are :

* Try UNet++ to deal with the absence of prediction of small tumors.
* Try methods with patching for this same issue.
* Implement and fine-tune different losses to increase sensibility of the UNet model. Obviously, this step can benefit to the other models (UNet++ and others) when implemented.

## unet-weighted-bce.ipynb : implementing the weighted binary crossentropy loss

#### A few observations :

Weighted BCE considerably improves the results! The previous version with class binary crossentropy predicted a blank mask when a tumor had to be found on the input. With weighted BCE, this doesn't happen. Moreover, this method seems to more precise as it seems to better handle edgy tumors.

The weigth used in this implementation is simply computed as the mean number of *1.* pixels in the trainning masks. This may be improved or at least require a little bit more of work.
