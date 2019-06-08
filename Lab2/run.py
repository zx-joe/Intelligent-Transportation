# import the necessary packages
import Softmax.data_utils as du
import argparse
import numpy as np
from Softmax.linear_classifier import Softmax 
from Pytorch.my_model import Net
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

def predict_usingPytorch(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    #pass
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    checkpoint = torch.load("Pytorch/model_vgg.ckpt")
    new_net = Net(None,None,None)
    new_net.load_state_dict(checkpoint)
    #outputs = new_model(X)
    #_ , y_pred = torch.max(F.softmax(outputs), 1)
    
    y_pred=np.array([])
    for images in X:
        outputs = new_net(images)
        _ , temp_y_pred = torch.max(F.softmax(outputs), 1)
        y_pred=np.append(y_pred,temp_y_pred)
    

    return y_pred
   
def predict_usingSoftmax(X):
    #########################################################################
    # TODO:                                                                 #
    # - Load your saved model                                               #
    # - Do the operation required to get the predictions                    #
    # - Return predictions in a numpy array                                 #
    #########################################################################
    #pass
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    
    with open('Softmax/softmax_weights.pkl', 'rb') as f:
        W = pickle.load(f)  
    new_softmax = Softmax()
    new_softmax.W = W.copy()
    y_pred = new_softmax.predict(X)
    return y_pred

def main(filename, group_number):
    
    X,Y = du.load_CIFAR_batch(filename)
    X = np.reshape(X, (X.shape[0], -1))
    mean_image = np.mean(X, axis = 0)
    X -= mean_image
    prediction_pytorch = predict_usingPytorch(X)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    prediction_softmax = predict_usingSoftmax(X)
    acc_softmax = sum(prediction_softmax == Y)/len(X)
    acc_pytorch = sum(prediction_pytorch == Y)/len(X)
    print("Group %s ... Softmax= %f ... Pytorch= %f"%(group_number, acc_softmax, acc_pytorch))
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test", required=True, help="path to test file")
    ap.add_argument("-g", "--group", required=True, help="group number")
    args = vars(ap.parse_args())
    main(args["test"],args["group"])