import numpy as np
import os
import logging


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

def group_node(node_edge_row:dict)->list:
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
        merage_list.append(set(group_set))


    print(merage_list)
    print(merage_list.__len__())










    pass





