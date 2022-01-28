# UNet4MRI

Using UNet and UNet++ to compute Brain MRI segmentation. Both networks are implemented in Tensorflow due to the current lack of available examples online (most of them are in PyTorch). 

## unet.ipynb

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
