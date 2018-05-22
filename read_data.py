from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
from python_pfm import readPFM
import numpy as np
import torch
class sceneDisp(Dataset):

    def __init__(self, root_dir,settype,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.settype=settype
        self.transform = transform
        finl = open('paths_left_'+settype+'.pkl','rb')
        finr = open('paths_right_'+settype+'.pkl', 'rb')
        self.paths_left = pickle.load(finl)
        self.paths_right = pickle.load(finr)
        finl.close()
        finr.close()
        finl = open('disp_left_'+settype+'.pkl', 'rb')
        finr = open('disp_right_'+settype+'.pkl', 'rb')
        self.disp_left = pickle.load(finl)
        self.disp_right = pickle.load(finr)
        finl.close()
        finr.close()
        for i in range(len(self.paths_left)): #to solve the inconsistency of index between rgb and depth
            a = self.paths_left[i].split('/')[-1].split('.')[0]
            b = self.disp_left[i].split('/')[-1]
            l=self.disp_left[i].replace(b, a + '.pfm')
            r=self.disp_right[i].replace(b, a + '.pfm')
            self.disp_left[i]=l
            self.disp_right[i]=r

    def __len__(self):
        if self.settype=='train':
            return 35454
        if self.settype=='test':
            return 4370

    def __getitem__(self, idx):

        # print(self.paths_left[idx])
        # print(self.paths_right[idx])
        # print(self.disp_left[idx])
        # print(self.disp_right[idx])
        imageL = cv2.imread(self.paths_left[idx]).reshape(540,960,3)#.transpose((2, 0, 1))
        imageR = cv2.imread(self.paths_right[idx]).reshape(540,960,3)#.transpose((2, 0, 1))
        dispL = readPFM(self.disp_left[idx])[0].astype(np.uint8).reshape(540,960,1).transpose((2, 0, 1))
        sample = {'imL': imageL, 'imR': imageR, 'dispL': dispL}
        if self.transform is not None:
            sample['imL']=self.transform(sample['imL'])
            sample['imR']=self.transform(sample['imR'])
        return sample

