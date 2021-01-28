#encoding=utf8
import torch
import numpy as np


def check_node_merge_node(node_list:list,mer_node:list):
    signle_node_loss = []
    for node in node_list:
        if node not in mer_node:
            signle_node_loss.append([node])
    if len(signle_node_loss)>0:
        is_los = True
    return signle_node_loss,is_los


def create(rows:list,cols:list,node_feature:list):
    '''

    :param rows: 合并后的每行[{},{},{},{}]
    :param col: 合并后的每列[{},{},{}]
    :param node_feature: [[],[],[]]
    :return:
    '''
    is_loss = 0 #0,1,2,3: 0 都不缺失，1 行缺失，2 列缺失，3 行列都缺失
    nodes_list = [i for i in range(len(node_feature))]

    #check 是否有丢失（在测试中发现有些孤立的数据莫名其妙的丢失；但是可能是行合并丢失，也有可能是列合并丢失，所以首先得进行node的校验）
    #1、check row
    c_row_l = [].extend(list(i) for i in rows)
    row_los,row_is_los = check_node_merge_node(nodes_list,c_row_l)
    if row_is_los:
        rows.extend(row_los)
    #2、check col
    c_clo_l = [].extend(list(i) for i in cols )
    col_los, col_is_los = check_node_merge_node(nodes_list, c_clo_l)
    if col_is_los:
        cols.extend(col_los)

    if row_is_los and col_is_los:
        is_loss = 3
    elif row_is_los and (not col_is_los):
        is_loss = 1
    elif col_is_los and (not row_is_los):
        is_loss = 2
    else:
        is_loss = 0

    #做个row的倒排索引
    row_dp_index = {}
    for i in nodes_list:
        i_dp = []
        for j in rows:
            if i in j:
                [].extend(j)
                i_dp.remove(i)
            row_dp_index[i] = list(set(i_dp))
    #merage:
    #1、先对行排序，取较大的那行，再对行内数据排序
    #2、再以此行为基础进行列添加
    #3、最后处理单个数据

    #first
    sort_list = {}
    idx = 0
    for i in rows:
        sort_list[idx] = len(i)
        idx += 1
    o_sort = sorted(sort_list.items(), key=lambda kv: (kv[1], kv[0]))
    o_sort  = o_sort[-1][0]
    start_row = rows[int(o_sort)] #以此作为开始的起始行
    #获取每个点的feature<row 是得到 x 坐标>
    start_row_x = {i:node_feature[int(i)][4] for i in start_row} #此处在node feature 中 node feature的 cell的中点X坐标

    start_row_x = sorted(start_row_x.items(), key=lambda kv: (kv[1], kv[0])) #起始序列已经排好序
    start_row_x = [i[0] for i in start_row_x]
    #[1,3,4,6,2,5]

    #占时先不这样干
    table_list = list() #做成二维数组

    #将 col 转化为dict 目的是为了标明下，方便后面看哪些数据没有加

    cols_dict = {}
    for i in range(len(cols)):
        c_dict = {j:node_feature[int(j)][5] for j in cols[i]} # 第五个位置是 y
        c_dict = sorted(c_dict.items(), key=lambda kv: (kv[1], kv[0]))
        col = [i[0] for i in c_dict]
        cols_dict[i] = col  #{0:[1,2,3],1:[2,3,4]}

    #上面对每一个列都进行了排序
    #开始构建

    # 1、 按照row中的位置对整个col排下序
    sor_col_idx = []
    for node in start_row_x:
        for i in range(len(cols)):
            if node in cols_dict[i]:
                sor_col_idx.append(i)
            pass

    diao c_table


    pass


def c_table(table:list,col:list,dp_dict:dict):
    c_dict = dict()
    if len(table) == 0:
        for i in range(len(col)):
            c_dict[str(0)+"_"+str(i)] = col[i]
        table.append(c_dict)
    else:
        t_c = len(table)
        col_1 = table[-1]
        c_d = {col[i]:i for i in range(len(col))}
        #用倒排check
        keys = c_d.keys()
        for i in col_1:
            i_list = dp_dict[i]
            idx = 0
            for k in keys:
                if i in i_list:
                    c_dict[str(t_c)+"_"+str(idx)] = k
                    break;
                else:
                    idx -= 1;

        table.append(c_dict)

    return table

        pass


    pass



if __name__ == "__main__":
    s = [1,3,8,6]
    s.sort()

    print(s)
    key_value = {}
    key_value[2] = 56
    key_value[1] = 2
    key_value[5] = 12
    key_value[4] = 24
    key_value[6] = 18
    key_value[3] = 323

    print("按值(value)排序:")
    s=sorted(key_value.items(), key=lambda kv: (kv[1], kv[0]))
    print(s)