'''
This is a config file of Text_Cnn
'''
import numpy as np
import torch

class DefaultConfig(object):
    
    data_root = ''
    train_data_root = ''
    test_data_root = ''
    eva_data_root = ''
    
    unk_vec = np.zeros(300,dtype=np.float32) #unknown vector
    pad_vec = np.zeros(300,dtype=np.float32) #padding vector
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use GPU if available
    filter_size = [3,4,5]        #filter size of CNN
    filter_num = [100,100,100]   #number of each filter size
    
    lr = 0.05           
    epochs = 20        
    batch_size = 2     
    
    print_freq = 100   # every 100 batchs print info
    


