from nets.models import nets
from nets.Residual import resblock
from nets.inception import inception
from nets.Resnext import resnext
from DeepLearning.training import train
from Dataloaders.dataload import train_dataset,test_dataset
from utils.yml import load_config
import matplotlib.pyplot as plt
from utils.plot import plot
parameters=load_config()
top1_acc_val,top1_acc_train,loss_avg_val,loss_avg_train=train(train_dataset,test_dataset,nets,resblock
,parameters['batch_size'],parameters['epochs'],parameters['learning_rate'])

plot(top1_acc_val,top1_acc_train,loss_avg_val,loss_avg_train,parameters['block'])