#encoding=utf-8
import networkx as nx
import requests
import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx


import matplotlib.pyplot as plt

from new_gfte.GFTE_new_pos.new_single_pos_dataset import GFTE_POS_DATASET
from new_gfte.GFTE_new_pos.new_single_pos_model import SinglePosNet


def get_pos_list_from_psenet(url,img_path):
    with open(img_path, 'rb') as fp:
        r = requests.post(url, files={"file": fp})
        output = r.json()
        bblist = output["bbx"]
        print(bblist)
        bblist = [[item[0][0], item[0][1], item[2][0], item[2][1]] for item in bblist]
        return bblist
        # print(bblist)
        # predata = PRED_DATASET(bblist)
        # print(predata[0].edge_index)
        # g = to_networkx(predata[0])
        # nx.draw(g)
        # plt.show()

def pos_pred(num_node_feature,num_out_class,base_channel,model_static_path,img_path,url):
    #pos_list = get_pos_list_from_psenet(url,img_path)

    #data =PRED_DATASET(pos_list)
    data = GFTE_POS_DATASET(r"F:\imgs\SciTSR\train",None,None)
    #data = DataLoader(dataset, batch_size=1, shuffle=False)
    pos_model = SinglePosNet(num_node_feature,num_out_class,base_channel)
    #pos_model = Net(num_node_feature, num_out_class)
    pos_model.load_state_dict(torch.load(model_static_path))
    y = pos_model(data[0])
    print(data[0].edge_index)
    print(y.argmax(dim=1))
    print(sum(y.argmax(dim=1)))

    #unsqueeze() 添加一个维度
    y1 = torch.cat((data[0].edge_index,(y.argmax(dim=1)).unsqueeze(0)),0)

    print(y1.T)







if __name__ == "__main__":

#F:\imgs\SciTSR\train\img\0705.0450v1.7.png
    path = r"F:\imgs\SciTSR\train\img\0810.1383v2.3.png"
    url = r"http://10.17.90.10:9000/predict"
    #model_static_path = r"F:\python\GFTE\models\pos_model\GFTE_POS_15_epoch_loss_0.3955_evaloss_0.3956_acc_0.8198.pth"
    #model_static_path =r"F:\python\GFTE\models\pos_model\GFTE_POS_100_epoch_loss_0.3573_evaloss_0.3590_acc_0.8311.pth"
    model_static_path = r"F:\python\GFTE\models\pos_model_mg\n_gfte_tloss_0.3613_eloss_0.3665_eacc_0.8300.pth"

    num_node_feature = 8
    num_out_class = 2
    base_channel = 32
    pos_pred(num_node_feature,num_out_class,base_channel,model_static_path,path,url)

