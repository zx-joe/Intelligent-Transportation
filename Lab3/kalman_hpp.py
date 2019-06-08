# import the necessary packages
import data_utils as du
import argparse
import numpy as np
from my_model_vgg import VGG
from my_model_resnet import ResNet
from my_model_resnet import BasicBlock
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.utils.data as utils
import time
import pdb
from torch.utils.data.sampler import SubsetRandomSampler
%matplotlib inline


#########################################################################
# TODO:                                                                 #
# This is used to input our test dataset to your model in order to      #
# calculate your accuracy                                               #
# Note: The input to the function is similar to the output of the method#
# "get_CIFAR10_data" found in the notebooks.                            #
#########################################################################

def predict_usingCNN(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    # Note: For the predictions, you have to return the index of the max    #
    # value                                                                 #
    #########################################################################
    #pass
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    
    # if we choose resnet, we can unlabel the following lines
    '''
    checkpoint = torch.load("model_resnet.ckpt")
    new_net = ResNet(BasicBlock, [3, 3, 3])        
    '''
    # the following 2 lines of code correspond to VGG net
    checkpoint = torch.load("model_vgg_32.ckpt")
    new_net = VGG()
    
    
    new_net.load_state_dict(checkpoint)
    y_pred=np.array([])
    for images in X:
        outputs = new_net(images)
        _ , temp_y_pred = torch.max(F.softmax(outputs), 1)
        y_pred=np.append(y_pred,temp_y_pred)
    

    return y_pred
   

def main(filename, group_number):
    X,Y = du.load_CIFAR_batch(filename)
    prediction_cnn = predict_usingCNN(X)
    acc_cnn = sum(prediction_cnn == Y)/len(X)
    print("Group %s ... CNN= %f"%(group_number, acc_cnn))
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])