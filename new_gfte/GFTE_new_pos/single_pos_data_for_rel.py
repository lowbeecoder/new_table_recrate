#encoding=utf-8

import os
import numpy as np
from torch_geometric.data import Dataset,Data
import torch
import json
import torch_geometric.transforms as GT
import logging

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import to_networkx


'''
by: billyx@synnex.com [lowbee_coder@sina.com]
原有的Dataset过于复杂，冗余太多，对于图卷积理解度不太友好
重写图卷积数据生成，data生成过程如下：
此处是基于pos的位置来一次pos各点之间的关系，实际上是基于edge_index做出预测 size(2,N),所以model最终预测出来的是edge_index
图Data构造流程：
Data的构造是基于图DATA的几要素构造<>
X:节点特征 [N,M] N个点，M维特征
edge_index：节点之间的边
POS：节点的位置(N,localtion)
Y:样本标签特征
edge_attr:节点边的特征（Nedge,m）
当然还有其他特征，data.face
但并不是每一个特征都会使用

GFTE使用的就是点pos特征
在data中通过knn得到边<edge_index>
再通过样本中的各点关系（colum,row）得到是否是同行/列，即为样本Y值

model训练实际上是为了确定表征edge_index是否能准确表征出
model的最后输出实际上是edge_index
最后通过pos,edge_index 得到图
'''

class GFTE_POS_DATASET(Dataset):
    '''
    1、root_path 应该精确到train/test目录
    2、直接加载json目录读取json list
    3、官方在dataset时加载了json,或生成了了imglist( dump到json) 此处摒弃
    '''
    def __init__(self,root_path,transform=None,pre_transform=None):
        super(GFTE_POS_DATASET,self).__init__(root_path,transform,pre_transform)
        self.root_path = root_path
        self.json_file_list = os.listdir(os.path.join(root_path,"structure"))
        #清洗数据，基于json 开始，通过json文件名判断是否有相应的img文件，再通过json与chunk判断数据的有效性和合法性，实际就是看是否json与chunk数据匹配
        self.imglist = os.listdir(os.path.join(root_path,"mytrain")) #img
        #此处是使用check后的list
        self.imglist = self.check_all()
        self.graph_transform = GT.KNNGraph(k=6) #使用pos创建k-nn图 Creates a k-NN graph based on node positions :obj:`pos`.


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

    def check_all(self):
        '''
        this function is use to check json文件和chunk文件中cell的标注是否为空，两者是否匹配
        :return: 返回的校验通过的 文件name
        '''
        vlidlist = list()
        for idx in range(len(self.imglist)):
            img = self.imglist[idx]
            logging.info("read img and info %s"%img)
            print("load file %s"%img)
            structs,chunk,_ = self.reallab(idx)

            check = self.check_chunks(structs,chunk,img)
            if check == 1:
                vlidlist.append(img)
        print("pass check the list is ",len(vlidlist))
        return vlidlist

    def reallab(self,idx):
        img_name = self.imglist[idx]
        img = os.path.join(self.root_path,"img",img_name) #img
        struct = os.path.join(self.root_path,"structure",os.path.splitext(os.path.basename(img_name))[0]+".json")
        chunk = os.path.join(self.root_path,"chunk",os.path.splitext(os.path.basename(img_name))[0]+".chunk")
        rel = os.path.join(self.root_path,"rel",os.path.splitext(os.path.basename(img_name))[0]+".rel")

        if not os.path.exists(struct) or not os.path.exists(chunk):
            logging.error("can't find struct or chunk on [%s]"%img_name)
            print("can't find cell info about %s"%img_name)
            return
        with open(struct,'r',encoding='utf8') as sf:
            structs = json.load(sf)['cells']
        with open(chunk,'r',encoding='utf8') as cf:
            chunks = json.load(cf)['chunks']
        with open (rel,'r',encoding='utf8') as rf:
            ref = rf.readlines()
        return structs,chunks,ref


    def check_chunks(self,structs,chunks,img):
        structs = self.remove_empty_cell(structs)
        idx = 0
        for st in structs:
            id = st['id']
            if len(structs)>len(chunks):
                logging.error("$s structs is not match chunks " + str(img))
                print(img," check error....")
                return 0
            chk = chunks[idx]
            text_stu = st['tex'].strip().replace(" ","")
            text_chu = chk['text'].strip().replace(" ","")
            content = "".join(st["content"])
            if text_chu != text_stu:
                if text_chu != content:
                    try:
                        print(img,">>",id," not match ----st_text: ",text_stu," ch_test: ",text_chu,"  content: ",content)
                    except:
                        print(img,">>",id," not match ---- 【】")
                    #return 0
            if st["end_row"]-st["start_row"] != 0 or st["end_col"]-st["start_col"] !=0:
                print("span cell ")
                if st["end_row"]-st["start_row"] != 0:
                    print("span row cell")
                if st["end_col"]-st["start_col"] !=0:
                    print("span cloum cell")
            idx += 1
        return 1

    def remove_empty_cell(self,structs):
       '''
       移除内容是空的cell,理论上cell内容不应该为空，即使为空那么也只能算是值为""
       过滤空的表格，也就是说对置空的cell此方法没有任何作用
       :param structs:
       :return:
       '''
       structs.sort(key=lambda p:p["id"]) #以ID排序
       news = []
       idx = 0
       for st in structs:
           text = st["tex"].strip().replace(" ","")
           content = len(st["content"])
           if (text==""or text=='$\\mathbf{}$') or content==0:  #空cell   # if text=="" or text=='$\\mathbf{}$': #空cell
               idx += 1
               continue
           st["id"] = idx
           news.append(st)
           idx += 1
       return news


    def __len__(self):
        return len(self.imglist)


    def cl_pos_limit(self,chunks):
        '''
        用于将获取文字位置(即整个表格上文字x的最大，最小，y的最大，最小)
        相当于得到了整个表格的长宽（左上，右下）
        :return:
        '''
        x_min = min(chunks,key=lambda p:p["pos"][0])["pos"][0]
        x_max = max(chunks,key=lambda p:p["pos"][1])["pos"][1]
        y_min = min(chunks,key=lambda p:p["pos"][2])["pos"][2]
        y_max = max(chunks,key=lambda p:p["pos"][3])["pos"][3]
        return [x_min,x_max,y_min,y_max,x_max-x_min,y_max-y_min]

    def pos_feater(self,chk,cl):
        '''
        得到每个cell的相对位置
        :param chk: 每个表格的xxyy(左上，右下)
        :param cl: table的左上，右下（长宽）
        :return:
        '''
        x_min = (chk["pos"][0])/cl[4]
        x_max = (chk["pos"][1])/cl[4]
        y_min = (chk["pos"][2])/cl[5]
        y_max = (chk["pos"][3])/cl[5]
        x_c = (x_min+x_max)/2.0
        y_c = (y_min+y_max)/2.0
        w = x_max-x_min
        h = y_max-y_min
        return [x_min,x_max,y_min,y_max,x_c,y_c,w,h]

    def cal_lab(self,data,tbpos):
        '''
        得到Y的lab(同行，同列) 相当于计算标注
        判断边同行，同列
        :param data:
        :param tbpos:
        :return:
        '''
        edge = data.edge_index #得到边[2,n]
        y = list()
        for i in range(edge.size()[1]):
            y.append(self.check_row(edge[0,i],edge[1,i],tbpos)) #check 是否同行
        return y

    def cal_rel_lab(self,data,tbpos,rel_list):

        edge = data.edge_index  # 得到边[2,n]
        #对rel-list 处理，做成{} key:p1-p2  value:1 or 0
        rel_dict = {}
        for rel in rel_list:
            rel_items = rel.split("	")
            row_col = rel_items[2].split(":")[0]
            rel_dict[str(rel_items[0]+"_"+str(rel_items[1]))] = 1 if row_col==str(1) else 2
            rel_dict[str(rel_items[1] + "_" + str(rel_items[0]))] = 1 if row_col == str(1) else 2
            y = []
        for i in range(edge.size()[1]):
            #y.append(self.check_row(edge[0, i], edge[1, i], tbpos))  # check 是否同行
            y.append(self.check_rel_row(edge[0, i],edge[1, i],tbpos,rel_dict))
        return y



    def check_rel_row(self,start_node,end_node,tbpos,rel_dict):
        # start_p1,start_p2 = tbpos[start_node][0],tbpos[start_node][1]
        # end_p1,end_p2 = tbpos[end_node][0],tbpos[end_node][1]
        key = str(start_node.item())+"_"+str(end_node.item())
        if key in rel_dict.keys():
            y = rel_dict[key] if rel_dict[key] == 1 else 0
        else:
            y = self.check_row(start_node,end_node,tbpos)
        return y


    def check_rel_colum(self,start_node,end_node,tbpos,rel_dict):
        key = str(start_node.item()) + "_" + str(end_node.item())
        if key in rel_dict.keys():
            y = 1 if rel_dict[key] == 2 else 0
        else:
            y=0
        return y


    def check_row(self,start,end,tbpos):
        st_start,st_end = tbpos[start][0],tbpos[start][1]
        ed_start,ed_end = tbpos[end][0],tbpos[end][1]
        if st_start>=ed_start and st_end<=ed_end:
            return 1
        if ed_start>=st_start and ed_end<=st_end:
            return 1
        return 0

    def check_colum(self,start,end,tbpos):
        st_start, st_end = tbpos[start][2], tbpos[start][3]
        ed_start, ed_end = tbpos[end][2], tbpos[end][3]
        if st_start >= ed_start and st_end <= ed_end:
            return 1
        if ed_start >= st_start and ed_end <= st_end:
            return 1
        return 0



    def get(self,idx):
        structs,chunks,rel_list = self.reallab(idx)

        cl_pos = self.cl_pos_limit(chunks) #得到整个表格的左上，右下；整个table长宽
        structs = self.remove_empty_cell(structs) #已经通过id排过序了
        x = list() #Data 的 X
        pos = list()
        tbpos = list()
        id = 0
        for st in structs:
            #id = st['id']
            chk = chunks[id]
            pos_feature = self.pos_feater(chk,cl_pos)
            x.append(pos_feature)
            pos.append(pos_feature[4:6]) #中心点与宽高
            #pos.append(pos_feature[:4]) #归一化坐标
            tbpos.append([st["start_row"],st["end_row"],st["start_col"],st["end_col"]])
            id += 1
        x = torch.tensor(x)
        pos = torch.FloatTensor(pos)
        data = Data(x=x,pos=pos)
        data = self.graph_transform(data) #构造图连接此时会生成edge_index


        y = self.cal_rel_lab(data,tbpos,rel_list)
        data.y = torch.LongTensor(y)
        return data




if __name__ == "__main__":
    '''
    每一个表是一个图数据
    '''


    # print(os.path.splitext(os.path.basename("xxxxx.text"))[0])
    # with open(r'F:\imgs\SciTSR\train\chunk\0704.2596v1.2.chunk',"r") as f:
    #     #print(json.load(f)["cells"])
    #     st = json.load(f)["chunks"]
    #     x_min = min(st, key=lambda p: p["pos"][0])["pos"][0]
    #     x_max = max(st, key=lambda p: p["pos"][1])["pos"][1]
    #     y_min = min(st, key=lambda p: p["pos"][2])["pos"][2]
    #     y_max = max(st, key=lambda p: p["pos"][3])["pos"][3]
    #     print(x_min,x_max,y_min,y_max)
    rootPath = r"F:\imgs\SciTSR\train"
    test = "Y"
    dataset = GFTE_POS_DATASET(rootPath,None,None)
    print(dataset[0])
    print(dataset[0].keys)
    print(dataset[0]['edge_index'])
    print(dataset[0]['y'])
    print("Y长度",len(dataset[0]['y']))
    print(sum(dataset[0]['y']))

    edge_index = dataset[0]['edge_index']
    pk = torch.cat((dataset[0].edge_index, (dataset[0]['y']).unsqueeze(0)), 0)
    edge_index = edge_index.t()
    print(edge_index)
    print(pk.T)

    for i in range (20):
        g = to_networkx(dataset[i])
        nx.draw(g)
        plt.show()

