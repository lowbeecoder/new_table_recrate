#encoding=utf-8
import os

import networkx as nx
from  torch_geometric.data import Dataset,Data
import json
import torch_geometric.transforms as GT
import logging
import requests
import cv2
import numpy as np
import torch
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt


class PRED_DATASET(Dataset):
    def __init__(self,pos_list:list):
        super(PRED_DATASET,self).__init__()
        self.graph_transform = GT.KNNGraph( k=6)
        self.pos_list = pos_list

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def reset(self):
        pass

    def read_structure(self):
        return

    def __len__(self):
        return 1

    def get_pos_limit(self):
        pos_List = self.pos_list
        pos_List = np.array(pos_List).reshape((-1,4))
        # x_min = min(pos_List[:,0])
        # x_max = max(pos_List[:,2])
        # y_min = min(pos_List[:,1])
        # y_max = max(pos_List[:,3])
        #为排除解析点换位子的问题，所以取值应该在所有x坐标取max或min
        x_min = min(np.append(pos_List[:,0],pos_List[:,2]))
        x_max = max(np.append(pos_List[:,0],pos_List[:,2]))
        y_min = min(np.append(pos_List[:,1],pos_List[:,3]))
        y_max = max(np.append(pos_List[:,1],pos_List[:,3]))
        w = abs(x_max-x_min)
        h = abs(y_max-y_min)
        return x_min,x_max,y_min,y_max,w,h

    def get_pos(self):
        _,_,_,_,w,h = self.get_pos_limit()
        pos_List = np.array(self.pos_list,np.float).reshape((-1,4))

        pos_List[:,0] = pos_List[:,0]/w
        pos_List[:, 2] = pos_List[:, 2] / w
        pos_List[:, 1] = pos_List[:, 1] / h
        pos_List[:, 3] = pos_List[:, 3] / h
        center_x = (pos_List[:,2]+pos_List[:,0])/2.0
        center_y = (pos_List[:,3]+pos_List[:,1])/2.0
        w = abs(pos_List[:,2]-pos_List[:,0])
        h = abs(pos_List[:,3]-pos_List[:,1])
        feature = np.array([center_x,center_y,w,h])
        features = np.concatenate((pos_List,feature.T),axis=1)
        return features

    def get(self,idx):
        features = self.get_pos()
        features = torch.FloatTensor(features)
        #features = features.double()
        pos = features[:,-4:]
        data = Data(x=features,pos=pos)
        data = self.graph_transform(data)
        return data


if __name__ == "__main__":

    # pos_List = [[1,2,4,6,],[3,4,7,8],[0,6,3,2],[1,4,7,3],[2,3,4,5]]
    #
    # os_List = np.array(pos_List).reshape((-1, 4))
    # # x_min = min(np.append(os_List[:, 0],os_List[:, 2]))
    # # x_max = max(os_List[:, 2])
    # # y_min = min(os_List[:, 1])
    # # y_max = max(os_List[:, 3])
    # #
    # # print(x_min, x_max, y_min, y_max)
    # center_x = (os_List[:,2] + os_List[:,0]) / 2.0
    # center_y = (os_List[:,3] + os_List[:,1]) / 2.0
    # w = abs(os_List[:,2] - os_List[:,0])
    # h = abs(os_List[:,3] - os_List[:,1])
    # nl = np.array([center_x,center_y,w,h])
    #
    # n = np.concatenate((os_List,nl.T),axis=1)
    # print(n)
    path = r"F:\imgs\SciTSR\test\img\0808.1125v1.5.png"
    url = r"http://10.17.90.10:9000/predict"
    with open(path,'rb') as fp:

        r = requests.post(url, files={"file": fp})
        output = r.json()
        bblist = output["bbx"]
        print(bblist)
        bblist = [[item[0][0],item[0][1],item[2][0],item[2][1]]for item in bblist]



        print(bblist)
        predata = PRED_DATASET(bblist)
        print(predata[0].edge_index)
        print(predata[0].x)
        g = to_networkx(predata[0])
        nx.draw(g)
        plt.show()







    pass



