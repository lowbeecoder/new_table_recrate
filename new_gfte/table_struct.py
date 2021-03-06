#encoding=utf8
import torch
import numpy as np


def check_node_merge_node(node_list:list,mer_node:list):
    signle_node_loss = []
    is_los = False
    for node in node_list:
        if str(node) not in mer_node:
            signle_node_loss.append([node])
    if len(signle_node_loss)>0:
        is_los = True
    return signle_node_loss,is_los


def create(rows:list,cols:list,node_feature:list):
    '''

    :param rows: 合并后的每行[{},{},{},{}]
    :param col: 合并后的每列[{},{},{}]
    :param node_feature: [[],[],[]] 就是每个node的特征信息
    :return:
    '''
    is_loss = 0 #0,1,2,3: 0 都不缺失，1 行缺失，2 列缺失，3 行列都缺失
    nodes_list = [i for i in range(len(node_feature))]

    #check 是否有丢失（在测试中发现有些孤立的数据莫名其妙的丢失；但是可能是行合并丢失，也有可能是列合并丢失，所以首先得进行node的校验）
    #1、check row
    c_row_l = []
    for i in rows:
        c_row_l.extend(list(i))


    row_los,row_is_los = check_node_merge_node(nodes_list,c_row_l)
    if row_is_los:
        rows.extend(row_los)
    #2、check col
    c_clo_l = []
    for i in cols:
        c_clo_l.extend(list(i))
    col_los, col_is_los = check_node_merge_node(nodes_list, c_clo_l)
    if col_is_los:
        cols.extend(col_los)

    #上面已经补充了缺失的node（比如某些孤立的点）
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
            if str(i) in j:
                i_dp.extend(list(j))
                #nodes_list.remove(i)
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

    sorted_col = {}#对col顺序恢复正常
    #对排序后的col按照列调整到正常顺序
    for k in range(len(sor_col_idx)):
        sorted_col[k] = cols_dict[sor_col_idx[k]]
        #调用
    #table_list = c_table(table_list,sor_col_idx,cols_dict)  #table_list {[1_1:node,1_2:node],[],[],[]}
    # #diao c_table
    # table = c_table()
    sor_col_idx = list(sorted_col.keys())

    maxcol_idx = {i:len(sorted_col[int(i)]) for i in sor_col_idx} #1:3,2:4  得到每一列的node个数
    col_max_idx = sorted(maxcol_idx.items(), key=lambda kv: (kv[1], kv[0]))[-1][0] #取最大那个的idx(以此为第一列，后面调整一下就可以)
    table = dict()
    col_m = dict()
    for i in range(len(cols_dict[col_max_idx])):
        col_m[str(i)+"_"+str(col_max_idx)]=cols_dict[col_max_idx][i]
    table[col_max_idx] = col_m   #table {1:{1_1:1,1_2:2},2:{},3:{}}

    sor_col_idx.remove(col_max_idx) #剔除已加入table那个最大的col

    for col_ids in sor_col_idx:
        table = c_table(table,col_ids,sorted_col,col_max_idx,row_dp_index)


    print(table)



def c_table(table:dict,col_idx:int,dp_dict:dict,bg_idx:int,dp:dict): #dp 倒排索引
    '''

    :param table:  表
    :param col_idx: 当前需要计算是哪行行那列的 idx
    :param dp_dict: 列的dict 通过这个得到列node
    :param bg_idx: 标准列的idx
    :param dp:  倒排索引
    :return:
    '''
    c_dict = dict()
    key = table.keys()
    if len(key) == 0:
        for i in range(len(dp_dict[col_idx])):
            c_dict[str(i)+"_"+str(col_idx)] = dp_dict[col_idx][i]
        table[col_idx]=c_dict
    else:

        bg_col = dp_dict[bg_idx] #标定列
        col = dp_dict[col_idx] #需要排的列

        s_mak = 0 #用于标记此列第几个与标准列开始同行
       # 1找到首次同步的位置（要确定的这个）
        tb_idx = 0 #首次位置
        bg_first = 0 #标定的首次位置
        for node_c in col:
            tb_f = False
            for bg_node in bg_col:
                if node_c in dp[int(bg_node)]:
                    tb_f = True
                    break
                bg_first += 1
            if tb_f:
                break
            tb_idx += 1


    if tb_idx >0:
        idx = [i for i in range(1,1+tb_idx)]
        idx.reverse()
        for id in range(len(idx)):
            c_node = col[id]
            c_dict[str(0-id) + "_" + str(col_idx)] = c_node
        for idx in range(tb_idx+1,len(col)):
            c_node = col[idx]
            is_flag = False #是否在里面（同步后）
            K = 0
            for bg_node_idx in range(len(bg_col)):
                bg_node = bg_col[bg_node_idx]
                if c_node in dp[bg_node]:
                    c_dict[str(bg_node_idx) + "_" + str(col_idx)] = c_node
                    K = bg_node_idx
                    break
                else:
                    is_flag = True #没有（可能 超过最长的底边）
            if is_flag: #超出的
                c_dict[str(K+1) + "_" + str(col_idx)] = bg_node_idx
                K += 1
    else:
        for node in col:
            for bg_idx in range(len(bg_col)):
                bg_node = bg_col[bg_idx]
                if node in dp[int(bg_node)]:
                    c_dict[str(bg_idx) + "_" + str(col_idx)] = node
                    break;
    # keys = c_dict.keys()
    # new_key = list(keys)
    # for item_idx in range(len(new_key)):
    #     idx = new_key[item_idx].split("_")
        table[col_idx] = c_dict


#上面没有处理出现负行的情况
#后面加上对出现负行的纠正



    return table


if __name__ == "__main__":
    # s = [1,3,8,6]
    # s.sort()
    #
    # print(s)
    # key_value = {}
    # key_value[2] = 56
    # key_value[1] = 2
    # key_value[5] = 12
    # key_value[4] = 24
    # key_value[6] = 18
    # key_value[3] = 323
    #
    # print("按值(value)排序:")
    # s=sorted(key_value.items(), key=lambda kv: (kv[1], kv[0]))
    # print(s)

    x = [{'2', '3', '1'}, {'5', '7', '6', '4'}]
    s = list()
    for i in x:
        a = list(i)
        s.extend(a)
    print(s)

    row = [{'2', '3', '1'}, {'5', '7', '6', '4'}, {'10', '9', '11', '8'}, {'15', '13', '14'}, {'17', '16', '19', '18'}, {'22', '20', '21', '23'}, {'26', '27', '25'}, {'30', '28', '29', '31'}, {'34', '32', '35', '33'}]
    col = [{'1', '21', '17', '33', '25', '9', '29', '13', '5'}, {'3', '23', '11', '15', '19', '35', '27', '7', '31'}, {'26', '6', '14', '2', '0', '12', '18', '22', '30', '34', '24', '10'}, {'8', '4', '28', '20', '32', '16'}]



    node = [[1.554282784461975, 1.6860287189483643, 3.550462484359741, 3.5851361751556396, 1.6201558113098145, 3.5677993297576904, 0.13174596428871155, 0.034673869609832764], [1.3118009567260742, 1.4515496492385864, 3.4672460556030273, 3.501919746398926, 1.3816752433776855, 3.4845829010009766, 0.13974875211715698, 0.034673869609832764], [1.5309747457504272, 1.6652028560638428, 3.4672460556030273, 3.501919746398926, 1.5980888605117798, 3.4845829010009766, 0.13422802090644836, 0.034673869609832764], [1.7703803777694702, 1.9027550220489502, 3.4672460556030273, 3.501919746398926, 1.836567759513855, 3.4845829010009766, 0.13237464427947998, 0.034673869609832764], [0.928507387638092, 1.2323379516601562, 3.3812520503997803, 3.4159257411956787, 1.0804226398468018, 3.3985888957977295, 0.3038305938243866, 0.034673869609832764], [1.339381217956543, 1.4239637851715088, 3.3812520503997803, 3.4159257411956787, 1.3816725015640259, 3.3985888957977295, 0.08458259701728821, 0.034673869609832764], [1.5557976961135864, 1.6403802633285522, 3.3812520503997803, 3.4159257411956787, 1.5980889797210693, 3.3985888957977295, 0.08458259701728821, 0.034673869609832764], [1.7611846923828125, 1.9119611978530884, 3.3812520503997803, 3.4159257411956787, 1.8365730047225952, 3.3985888957977295, 0.15077650547027588, 0.034673869609832764], [0.928507387638092, 1.2323379516601562, 3.295264959335327, 3.3299388885498047, 1.0804226398468018, 3.3126020431518555, 0.3038305938243866, 0.034673869609832764], [1.339381217956543, 1.4239637851715088, 3.295264959335327, 3.3299388885498047, 1.3816725015640259, 3.3126020431518555, 0.08458259701728821, 0.034673869609832764], [1.5557976961135864, 1.6403802633285522, 3.295264959335327, 3.3299388885498047, 1.5980889797210693, 3.3126020431518555, 0.08458259701728821, 0.034673869609832764], [1.7611846923828125, 1.9119611978530884, 3.295264959335327, 3.3299388885498047, 1.8365730047225952, 3.3126020431518555, 0.15077650547027588, 0.034673869609832764], [1.535253643989563, 1.7050611972808838, 3.1953980922698975, 3.230072021484375, 1.6201574802398682, 3.2127349376678467, 0.1698075532913208, 0.034673869609832764], [1.3118009567260742, 1.4515496492385864, 3.1121816635131836, 3.146855354309082, 1.3816752433776855, 3.129518508911133, 0.13974875211715698, 0.034673869609832764], [1.5309747457504272, 1.6652028560638428, 3.1121816635131836, 3.146855354309082, 1.5980888605117798, 3.129518508911133, 0.13422802090644836, 0.034673869609832764], [1.7703803777694702, 1.9027550220489502, 3.1121816635131836, 3.146855354309082, 1.836567759513855, 3.129518508911133, 0.13237464427947998, 0.034673869609832764], [0.928507387638092, 1.2323379516601562, 3.0261945724487305, 3.060868501663208, 1.0804226398468018, 3.043531656265259, 0.3038305938243866, 0.034673869609832764], [1.322830319404602, 1.4405097961425781, 3.0261945724487305, 3.060868501663208, 1.3816701173782349, 3.043531656265259, 0.11767955124378204, 0.034673869609832764], [1.5557976961135864, 1.6403802633285522, 3.0261945724487305, 3.060868501663208, 1.5980889797210693, 3.043531656265259, 0.08458259701728821, 0.034673869609832764], [1.7446339130401611, 1.9285074472427368, 3.0261945724487305, 3.060868501663208, 1.8365706205368042, 3.043531656265259, 0.18387345969676971, 0.034673869609832764], [0.928507387638092, 1.2323379516601562, 2.9402005672454834, 2.974874496459961, 1.0804226398468018, 2.9575376510620117, 0.3038305938243866, 0.034673869609832764], [1.322830319404602, 1.4405097961425781, 2.9402005672454834, 2.974874496459961, 1.3816701173782349, 2.9575376510620117, 0.11767955124378204, 0.034673869609832764], [1.539246916770935, 1.6569263935089111, 2.9402005672454834, 2.974874496459961, 1.5980865955352783, 2.9575376510620117, 0.11767955124378204, 0.034673869609832764], [1.7446339130401611, 1.9285074472427368, 2.9402005672454834, 2.974874496459961, 1.8365706205368042, 2.9575376510620117, 0.18387345969676971, 0.034673869609832764], [1.561179518699646, 1.6791305541992188, 2.8403406143188477, 2.875014543533325, 1.6201549768447876, 2.857677698135376, 0.11795105040073395, 0.034673869609832764], [1.3118009567260742, 1.4515496492385864, 2.757124185562134, 2.7917981147766113, 1.3816752433776855, 2.774461269378662, 0.13974875211715698, 0.034673869609832764], [1.5309747457504272, 1.6652028560638428, 2.757124185562134, 2.7917981147766113, 1.5980888605117798, 2.774461269378662, 0.13422802090644836, 0.034673869609832764], [1.7703803777694702, 1.9027550220489502, 2.757124185562134, 2.7917981147766113, 1.836567759513855, 2.774461269378662, 0.13237464427947998, 0.034673869609832764], [0.928507387638092, 1.2323379516601562, 2.6711301803588867, 2.7058041095733643, 1.0804226398468018, 2.688467264175415, 0.3038305938243866, 0.034673869609832764], [1.322830319404602, 1.4405097961425781, 2.6711301803588867, 2.7058041095733643, 1.3816701173782349, 2.688467264175415, 0.11767955124378204, 0.034673869609832764], [1.5557976961135864, 1.6403802633285522, 2.6711301803588867, 2.7058041095733643, 1.5980889797210693, 2.688467264175415, 0.08458259701728821, 0.034673869609832764], [1.7446339130401611, 1.9285074472427368, 2.6711301803588867, 2.7058041095733643, 1.8365706205368042, 2.688467264175415, 0.18387345969676971, 0.034673869609832764], [0.928507387638092, 1.2323379516601562, 2.5851361751556396, 2.619810104370117, 1.0804226398468018, 2.602473258972168, 0.3038305938243866, 0.034673869609832764], [1.322830319404602, 1.4405097961425781, 2.5851361751556396, 2.619810104370117, 1.3816701173782349, 2.602473258972168, 0.11767955124378204, 0.034673869609832764], [1.539246916770935, 1.6569263935089111, 2.5851361751556396, 2.619810104370117, 1.5980865955352783, 2.602473258972168, 0.11767955124378204, 0.034673869609832764], [1.7446339130401611, 1.9285074472427368, 2.5851361751556396, 2.619810104370117, 1.8365706205368042, 2.602473258972168, 0.18387345969676971, 0.034673869609832764]]
    table = create(row,col,node)
    print(table)