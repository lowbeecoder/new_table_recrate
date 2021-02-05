#encoding=utf8

import numpy as np
import os
import logging
import torch
from collections import Counter


def get_same_rows_dict(data,y):
    edge = data[0].edge_index
    edge_len = len(edge[0])
    node_edge_dict_row = {} #所有测试为同行
    node_edge_same_row = {} #使双向关系转成单向关系
    for i in range(edge_len):
        keys = str((edge[0][i]).item()) + "_"+str((edge[1][i]).item())
        #revies_keys = str((edge[1][i]).item()) + "_"+str((edge[0][i]).item())
        if y[i] == 1:
            node_edge_dict_row[keys] = 1

    keys = node_edge_dict_row.keys()
    for key in keys:
        re_key = key.split("_")[1]+"_"+key.split("_")[0]
        if re_key in node_edge_dict_row:
            if node_edge_dict_row[key] != node_edge_dict_row[re_key]:
                logging.error(key,' is not equals ',re_key)
            else:
                if key not in node_edge_same_row and re_key not in node_edge_same_row:
                    node_edge_same_row[key] = 1
        else:
            node_edge_same_row[key] = 1

    return node_edge_same_row


def check_set_merage(set1,set2):
    for item in set1:
        if item in set2:
            return True
    return False

def group_node(node_edge_row:dict,node_feature:list,row_col)->list:
    keys = list(node_edge_row.keys())
    group_row = []
    while(len(keys)>0):

        f_key  = keys[0] #以key的首点为start点
        key_len = len(keys)

        s_k_list = f_key.split("_")
        f_set = set(s_k_list)
        del_key = []
        del_key.append(f_key)
        for i in range(1,key_len):
            k = keys[i]
            if (k.split("_")[0] in f_set) or (k.split("_")[1] in f_set):
                f_set.add(k.split("_")[0])
                f_set.add(k.split("_")[1])
                del_key.append(k)
        for k in del_key:
            keys.remove(k)
        group_row.append(f_set)
    merage_list = []
    while len(group_row)>0:
        g_len = len(group_row)
        del_list = [group_row[0]]
        group_set = list(i for i in list(group_row[0]))
        for row1 in range(1, g_len):
            flag = check_set_merage(group_set, group_row[row1])
            if flag:
                group_set.extend([i for i in list(group_row[row1])])
                del_list.append(group_row[row1])
        for i in del_list:
            group_row.remove(i)
        # row_list = list()
        # for gs in group_set:
        #     node_o_id = gs[0]
        #     o_set = set()
        #     o_set.add(node_o_id)
        #     sp_set = set()
        #     for node_id in gs:
        #         if abs(node_feature[node_id][5]-node_feature[node_o_id][5])<(node_feature[node_o_id][7])/2:
        #             o_set.add(node_id)
        #         else:
        #             sp_set
        #
        #     pass
        merage_list.append(set(group_set))

        print(merage_list)

        final_list = list()
        for set_group in merage_list:
            g_list = list()
            node_split(set_group,node_feature,g_list,row_col)
            final_list.extend(g_list)


    print(final_list)
    print(final_list.__len__())

#有合并的会出现不同行的变成同行的，这样必须通过此函数将其分割
def node_split(node_set,node_feature,split_list,row_col):
    node_list = list(node_set)
    set_o = set()
    set_m = set()
    node_o = node_list[0]
    set_o.add(node_o)
    for node_idx in node_list:

        if row_col:
            if abs(node_feature[int(node_idx)][5] - node_feature[int(node_o)][5]) < (node_feature[int(node_o)][7]) / 2:
                set_o.add(node_idx)
            else:
                set_m.add(node_idx)
        else:
            if abs(node_feature[int(node_idx)][4] - node_feature[int(node_o)][4]) < (max(node_feature[int(node_o)][6],node_feature[int(node_idx)][6])) / 2:
                set_o.add(node_idx)
            else:
                set_m.add(node_idx)
    split_list.append(set_o)
    if len(set_m) >1:
        node_split(set_m,node_feature,split_list,row_col)
    else:
        if len(set_m) != 0:
            split_list.append(set_m)





def merage(row_dict:list,colum_dict:list,x:list):
    cnt_rows = max([len(item) for item in row_dict])
    cnt_columns = max([len(item) for item in colum_dict])

    #随机取row最长的一行，再随机取colum最长的一列 （基于里面的交点来生长）
    max_row = None
    max_col = None
    for row_item in row_dict:
        if cnt_rows == len(row_item):
            max_row = row_item
            break
    for col_item in colum_dict:
        if cnt_columns == len(col_item):
            max_col = col_item
            break

    c_row_dict = {x[int(row_idx)][4]:int(row_idx) for row_idx in max_row}
    c_col_dict = {x[int(col_idx)][5]: int(col_idx) for col_idx in max_col}
    c_row_dict = sorted(c_row_dict.items(), key=lambda d: d[0])
    c_col_dict = sorted(c_col_dict.items(), key=lambda d: d[0])

    # if c_row_dict[0]>c_row_dict[1]:
    #     c_row_dict = c_row_dict.reverse()
    # if c_col_dict[0]>c_col_dict[1]:
    #     c_col_dict = c_col_dict.reverse()

    c_row = [item[1] for item in c_row_dict ]
    c_col = [item[1] for item in c_col_dict ]
    c_col.reverse()

    #col方向上最小那个必为最顶（第一排那个）<不一定> 【首先还是得处理缺失点】
    table_col_idx = 0
    for i in c_col:
        pass

#如果坐标系以左下为原点（那么y需要反转）
#如果坐标系以左上为原点（y不需要反转）
    # c_row = [item[1] for item in c_row_dict ]
    # c_col = [item[1] for item in c_col_dict ]
    # c_col.reverse()
    # k = 0
    # c_k_row_idx = 0
    # c_k_col_idx = 0
    # #先求两者的相同元素 和相应位置的index
    # for i in c_row:
    #     if i in c_col:
    #         k=i
    #         break
    #     c_k_row_idx += 1
    # for i in c_col:
    #     if i == k:
    #         break
    #     c_k_col_idx += 1
    #
    # #以列开始搜索
    # #while 1:
    # #    pass
    #
    # for col_idx in c_col:
    #     for
    print(c_row,c_col)



def check_row(row_dict:dict):

    pass




def table_r(row_dict:dict,colum_dict:dict,x:list):
    node_dict = dict()
    for i in range(len(x)):
        node_dict[i] = x[i]
    row_keys = list(row_dict.keys())
    col_keys = list(colum_dict.keys())
    row_node_set = set()
    col_node_set = set()
    #获取一个row的node
    n_n_list = list()
    for n_n in row_keys:
        n_n_list.extend(n_n.split("-"))
    node = Counter(n_n_list).most_common(1) #取出现最多的那个node
    #在进行从建时必须比较node的x,y坐标
    org_node = set ()
    del_node = set()
    x,y,w,h = node_dict[int(node[0][0])][4:] #中心点的x，y,以及长，宽
    for row in row_keys:
        org_node.add(int(node[0][0]))
        if str(node[0][0]) in [str(i) for i in row.split("_")]:
            idx_list = [int(i) for i in row.split("_")]
            idx_list.remove(int(node[0][0]))
            add_node = int(idx_list[0])
            x_a, y_a, w_a, h_a = node_dict[add_node][4:]
            if abs(y-y_a)<abs(h+h_a)/2:
                org_node.add(add_node)
    pass





if __name__ == "__main__":

    a = {'1_1','1_0','2_1','0_7','2_6'}
    al = list()
    for b in a:
        al.extend(b.split("_"))
    so = Counter(al).most_common(1)
    print(type(so))
    print(so)



    row =  [{'5', '6', '0', '3', '1', '4', '2','13', '7', '8', '9', '12', '11', '10'}, {'15', '17', '18', '16', '20', '14', '19'}, {'22', '26', '25', '23', '24', '21', '27','28', '33', '31', '34', '30', '32', '29','36', '35'}, {'39', '38'}, {'41', '42'}]
    column = [{'14', '41', '21', '7', '38', '35', '0', '28'}, {'29', '36', '22', '1', '39', '42', '8', '15'}, {'2', '23', '30', '9', '16'}, {'31', '3', '10', '17', '24'}, {'11', '32', '18', '25', '4'}, {'37', '26', '43', '12', '33', '5', '40', '19'}, {'6', '20', '27', '34', '13'}]

    x = [[0.5116, 0.5373, 5.2917, 5.3461, 0.5244, 5.3189, 0.0257, 0.0545],
        [0.6467, 0.6657, 5.2917, 5.3461, 0.6562, 5.3189, 0.0191, 0.0545],
        [0.7451, 0.8027, 5.2753, 5.3461, 0.7739, 5.3107, 0.0576, 0.0708],
        [0.8456, 0.9650, 5.2753, 5.3461, 0.9053, 5.3107, 0.1194, 0.0708],
        [1.0079, 1.1482, 5.2753, 5.3461, 1.0781, 5.3107, 0.1403, 0.0708],
        [1.2091, 1.2839, 5.2753, 5.3461, 1.2465, 5.3107, 0.0748, 0.0708],
        [1.3448, 1.4815, 5.2753, 5.3461, 1.4132, 5.3107, 0.1367, 0.0708],
        [0.5073, 0.5416, 5.1566, 5.2111, 0.5244, 5.1838, 0.0343, 0.0545],
        [0.6343, 0.6781, 5.1566, 5.2111, 0.6562, 5.1838, 0.0439, 0.0545],
        [0.7528, 0.7967, 5.1566, 5.2111, 0.7747, 5.1838, 0.0439, 0.0545],
        [0.8842, 0.9281, 5.1566, 5.2111, 0.9062, 5.1838, 0.0439, 0.0545],
        [1.0570, 1.1009, 5.1566, 5.2111, 1.0789, 5.1838, 0.0439, 0.0545],
        [1.1997, 1.2951, 5.1566, 5.2111, 1.2474, 5.1838, 0.0954, 0.0545],
        [1.3664, 1.4617, 5.1566, 5.2111, 1.4140, 5.1838, 0.0954, 0.0545],
        [0.5073, 0.5416, 5.0215, 5.0760, 0.5244, 5.0488, 0.0343, 0.0545],
        [0.6343, 0.6781, 5.0215, 5.0760, 0.6562, 5.0488, 0.0439, 0.0545],
        [0.7528, 0.7967, 5.0215, 5.0760, 0.7747, 5.0488, 0.0439, 0.0545],
        [0.8842, 0.9281, 5.0215, 5.0760, 0.9062, 5.0488, 0.0439, 0.0545],
        [1.0570, 1.1009, 5.0215, 5.0760, 1.0789, 5.0488, 0.0439, 0.0545],
        [1.1997, 1.2951, 5.0215, 5.0760, 1.2474, 5.0488, 0.0954, 0.0545],
        [1.3664, 1.4617, 5.0215, 5.0760, 1.4140, 5.0488, 0.0954, 0.0545],
        [0.5073, 0.5416, 4.8864, 4.9409, 0.5244, 4.9137, 0.0343, 0.0545],
        [0.6343, 0.6781, 4.8864, 4.9409, 0.6562, 4.9137, 0.0439, 0.0545],
        [0.7528, 0.7967, 4.8864, 4.9409, 0.7747, 4.9137, 0.0439, 0.0545],
        [0.8842, 0.9281, 4.8864, 4.9409, 0.9062, 4.9137, 0.0439, 0.0545],
        [1.0570, 1.1009, 4.8864, 4.9409, 1.0789, 4.9137, 0.0439, 0.0545],
        [1.1997, 1.2951, 4.8864, 4.9409, 1.2474, 4.9137, 0.0954, 0.0545],
        [1.3664, 1.4617, 4.8864, 4.9409, 1.4140, 4.9137, 0.0954, 0.0545],
        [0.4987, 0.5502, 4.7514, 4.8058, 0.5244, 4.7786, 0.0515, 0.0545],
        [0.6257, 0.6867, 4.7514, 4.8058, 0.6562, 4.7786, 0.0610, 0.0545],
        [0.7528, 0.7967, 4.7514, 4.8058, 0.7747, 4.7786, 0.0439, 0.0545],
        [0.8842, 0.9281, 4.7514, 4.8058, 0.9062, 4.7786, 0.0439, 0.0545],
        [1.0570, 1.1009, 4.7514, 4.8058, 1.0789, 4.7786, 0.0439, 0.0545],
        [1.1997, 1.2951, 4.7514, 4.8058, 1.2474, 4.7786, 0.0954, 0.0545],
        [1.3664, 1.4617, 4.7514, 4.8058, 1.4140, 4.7786, 0.0954, 0.0545],
        [0.4901, 0.5588, 4.6163, 4.6708, 0.5244, 4.6435, 0.0687, 0.0545],
        [0.6085, 0.7039, 4.6163, 4.6708, 0.6562, 4.6435, 0.0954, 0.0545],
        [1.1911, 1.3036, 4.6163, 4.6708, 1.2474, 4.6435, 0.1125, 0.0545],
        [0.4815, 0.5673, 4.4812, 4.5357, 0.5244, 4.5084, 0.0858, 0.0545],
        [0.6085, 0.7039, 4.4812, 4.5357, 0.6562, 4.5084, 0.0954, 0.0545],
        [1.1911, 1.3036, 4.4812, 4.5357, 1.2474, 4.5084, 0.1125, 0.0545],
        [0.4815, 0.5673, 4.3461, 4.4006, 0.5244, 4.3734, 0.0858, 0.0545],
        [0.6085, 0.7039, 4.3461, 4.4006, 0.6562, 4.3734, 0.0954, 0.0545],
        [1.1911, 1.3036, 4.3461, 4.4006, 1.2474, 4.3734, 0.1125, 0.0545]]
    x = torch.tensor(x)
    x = x.tolist()
    llist = list()

    node_split(row[0],x,llist)
    print(llist)



    #
    # merage(row,column,x)
    #
    #
    # pass







    pass





