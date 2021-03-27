
# Building the list_file for our Dataset.
"""
We have dataset such as UCF. It contains three splits (training on different training_set and test_set) then copmpute the average for 
better accuracy. Using these splits, we can access the dataset to build our list file which contains (directory of each video,
number of frames and its label)

First, we import some important libraries.
"""

import os                         #library to interact with your OS whether it is Windows, Linux or MAC                       
import glob                       #library used with os to access all the videos at the same time
import random                     #random number generator
import argparse

parser = argparse.ArgumentParser(description="List File Generation")
parser.add_argument('dataset_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('textfiles_dir', type=str)
args = parser.parse_args()

"""
Now, we will define two functions: one for extracting name and label for each video and the other for extracting each video directory
and number of frames.
"""

def SplitsInfoExtract1(textfiles_dir):
    """
    Extract name&label for each video 
    output: list of tuples (each tuple has trainlist and testlist) of list of tuple (each tuple has name of the video and its label)
    Note: we have three splits for training and testing
    """
    actionLabel = [x.strip().split() for x in open(os.path.join(textfiles_dir,'classInd.txt'))]  #[[1,'label1'],.....]
    actionLabel_dic = {x[1]:int(x[0])-1 for x in actionLabel}                  #{'label1':0, 'label2':1 ,...}
    
    
    
    def ExtractInfo(line):
        """
        Input: line form testlist or trainlist (eg : ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi )
        Output: the name and the label for this video
        """
        line = line.split('/')
        name = line[1].split('.')[0]
        label = actionLabel_dic[line[0]]
        return name,label
    
    Name_Label = []
    
    for i in range(1,4): #looping through the dataset splits to Extract information
        trainlist = [ExtractInfo(x) for x in open (os.path.join(textfiles_dir,'trainlist{:02d}.txt'.format(i)))] #Extract info from every video in the trian splits
        testlist  = [ExtractInfo(x) for x in open (os.path.join(textfiles_dir,'testlist{:02d}.txt'.format(i)))]  #Extract info from every video in the test splits
        Name_Label.append((trainlist,testlist))

    return Name_Label

def SplitsInfoExtract2(dataset_dir):
    '''
    Input: dataset directory 
    Output:RGB_count ---> number of frames in each video stored in a dictionary
           Video_dir ---> the directory of each video stored in a dictionary
    '''
    Framefiles_dir = glob.glob(os.path.join(dataset_dir,'*'))
    
    RGB_count = {}
    Video_dir = {}
    for file_dir in Framefiles_dir:
        Video_name = file_dir.split('/')[-1]   #WARNING:The splitor sign (/) may be different from a machine to another
        Frames_list = os.listdir(file_dir)
        RGB_count[Video_name] = len(Frames_list)
        Video_dir[Video_name] = file_dir
        
    return RGB_count, Video_dir

"""
We've built our two main functions for extracting information from each split. Now, we should merge them into one function that will
be used to generate our list file.
"""

def MergeInfo(Name_Label,Frames_dir, split_idx, shuffle=False):
    '''
    Inputs
        Name_Label : The output of SpiltsInfoExtract1
        Frames_dir : The output of SpiltsInfoExtract2
        split_inx : 1 to 4 (split number)
    Outputs:
        Train_DFL : Huge string every line of it consist of [Dirctory of the video -- number of frames -- label]
        Test_DFL  : Huge string every line of it consist of [Dirctory of the video -- number of frames -- label]
    '''
    Name_Label = Name_Label[split_idx-1]                               #Specify which split being processed (output: tuple(trainlist,testlist)
    train_info =  Name_Label[0]                                        #List of tuples each tuple is (name,label)
    test_info  =  Name_Label[1]                                        #List of tuples each tuple is (name,label)
    
    def DFL (Name_Label):                                                         #DFL : Directory , Frames , Label
        RGB_list = []
        for name_label in Name_Label:                                             #For each video in the split
            Video_dir = Frames_dir[1][name_label[0]]
            RGB_count = Frames_dir[0][name_label[0]]
            Label     = name_label[1]
            RGB_list.append('{} {} {}\n'.format(Video_dir, RGB_count, Label))     #packing variables into string -huge one-

            if shuffle:
                random.shuffle(RGB_list)
        return RGB_list
    
    Train_DFL = DFL(train_info)
    Test_DFL  = DFL(test_info)
    
    return Train_DFL,Test_DFL

"""
Now let's build our list file function that will be used for different datasets to generate directory, number of frames and label 
for each video.
"""

def Build_List_File(dataset_dir, out_dir, textfiles_dir, splits_num=1, shuffle=False):
    
    """
    Inputs:
        frames_dir: directory for the frames to be processed (one video at a time)
        out_dir: directory where the list_file will be generated
        splits_num: number of dataset splits (we will go with 1 split for simplicity)
        shuffle: True or False
    """
    Name_Label = SplitsInfoExtract1(textfiles_dir)
    Frames_dir = SplitsInfoExtract2(dataset_dir)
    
    for i in range(splits_num):
        Train_DFL,Test_DFL = MergeInfo(Name_Label,Frames_dir, i+1, shuffle)
        open(os.path.join(out_dir, 'rgb_train_FileList{}.txt'.format(i+1)), 'w').writelines(Train_DFL)
        open(os.path.join(out_dir, 'rgb_test_FileList{}.txt'.format(i+1)), 'w').writelines(Test_DFL)

"""Now, you can generate your own list file which you will find in the directory you will specifiy in out_dir parameter."""

if __name__=="__main__":
    Build_List_File(args.dataset_dir, args.output_dir, args.textfiles_dir, splits_num=1, shuffle=False)
    
    
    
    
