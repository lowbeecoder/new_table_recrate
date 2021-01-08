#encoding=utf-8
from collections import OrderedDict

import torch
from torch.nn import Linear,Module,Sequential,ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class SinglePosNet(Module):
    '''
    num_node_feature:node点的维度（node的channel）
    num_out_class:分类类数
    '''
    def __init__(self,num_node_feature,num_out_class,base_channel=32):
        super(SinglePosNet,self).__init__()
        self.num_node_feature = num_node_feature
        self.num_out_class = num_out_class
        self.base_channel = base_channel
        self.gcn_conv1 = GCNConv(self.num_node_feature,2*self.base_channel)
        self.gcn_conv2 = GCNConv(2*self.base_channel,2*self.base_channel)
        self.liner_model = self.build_line_model()

    def build_line_model(self):

        line_model =  Sequential(OrderedDict([
            ('liner1', Linear(self.base_channel * 4, self.base_channel * 2)),
            ('line_relu1',ReLU()),
            # ('liner2',Linear(self.base_channel*2,self.base_channel*1)),
            # ('line_relu2', ReLU()),
            ('liner3',Linear(self.base_channel*2,self.num_out_class)),
        ]))
        return line_model



    # def build_model(self):
    #     pos_model = Sequential(OrderedDict([
    #         ('gcn_conv1',GCNConv(self.num_node_feature,2*self.base_channel)),
    #         ('relu1',ReLU()),
    #         ('gcn_conv2',GCNConv(2*self.base_channel,2*self.base_channel)),
    #         ('relu2',ReLU())
    #         ('liner1',Linear(self.base_channel*4,self.base_channel*2))
    #         ('liner2',Linear(self.base_channel*2,self.base_channel*1))
    #         ('liner3',Linear(self.base_channel*1,self.num_out_class))
    #     ]))
    #     return pos_model

    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        x = self.gcn_conv1(x,edge_index)
        x = F.relu(x)
        x = self.gcn_conv2(x,edge_index)
        x = F.relu(x)
        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        x_pos = torch.cat((x1,x2),dim=1)
        x_pos = self.liner_model(x_pos)
        out = F.log_softmax(x_pos,dim=1)
        return out



class SinglePosNet_MG(Module):

    def __init__(self,node_feature,out_class):
        super(SinglePosNet_MG,self).__init__()
        self.node_feature = node_feature
        self.out_class = out_class

        self.g_cov1 = GCNConv(node_feature,128)
        self.g_cov2 = GCNConv(128,128)
        # self.g_cov3 = GCNConv(64,64)
        self.liner_mg = self.build_line_model_mg()

    def build_line_model_mg(self):
        line_model = Sequential(OrderedDict([
                ('liner1', Linear(128 * 2, 128)),
                ('line_relu1', ReLU()),
                # ('liner2', Linear(128, 64)),
                # ('line_relu2', ReLU()),
                ('liner3', Linear(128, self.out_class)),
            ]))
        return line_model

    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        x = self.g_cov1(x, edge_index)
        x = F.relu(x)
        x = self.g_cov2(x, edge_index)
        x = F.relu(x)
        # x = self.g_cov3(x, edge_index)
        # x = F.relu(x)
        x1 = x[edge_index[0]]
        x2 = x[edge_index[1]]
        x_pos = torch.cat((x1, x2), dim=1)
        x_pos = self.liner_mg(x_pos)
        out = F.log_softmax(x_pos, dim=1)
        return out
