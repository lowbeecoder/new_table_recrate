#encoding=utf-8
import torch_geometric
import torch
import os
import torch.optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import DataLoader

from new_gfte.GFTE_new_pos.new_single_pos_dataset import GFTE_POS_DATASET
from new_gfte.GFTE_new_pos.new_single_pos_model import SinglePosNet


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train_init(params,lr=1e-3,beta=0.5,step_size=20,gamma=0.9):

    optimizer = Adam(params, lr=lr, betas=(beta, 0.999), eps=1e-8,
                 weight_decay=0)
    criterion = torch.nn.NLLLoss()
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    return optimizer,criterion,scheduler

def train_process(model,epochs,train_data_set,eval_data_set,optimizer, criterion, scheduler,eval_step,store_path):

    for epoch in range(1,epochs+1):
        for para in model.parameters():
            para.requires_grad = True
        model.train()
        batch_total = len(train_data_set)
        loss_total = 0;
        for data in train_data_set:
            model.zero_grad()
            target = data.y
            pred = model(data)
            loss = criterion(pred,target)
            loss_total += loss.item()
            loss.backward()
            optimizer.step()
        t_loss = loss_total/float(batch_total)
        print("%d/%d train loss: %.4f"%(epoch,epochs,t_loss))
        scheduler.step()
        if(epoch%eval_step==0):
            model.eval()
            for para in model.parameters():
                para.requires_grad = False
            acc = 0
            eval_loss = 0
            n_total = 0
            eval_data_len = len(eval_data_set)
            for edata in eval_data_set:
                y = edata.y
                e_pred = model(edata)
                e_loss = criterion(e_pred,y)
                eval_loss+=e_loss
                label = edata.y.detach().numpy()
                _, e_pred = e_pred.max(1)
                preds = e_pred.detach().numpy()

                acc = acc + (label == preds).sum()
                n_total = n_total + label.shape[0]
            accuracy = acc / float(n_total)
            e_loss = eval_loss/float(eval_data_len)
            print("%d/%d train_loss:%.4f  eval_loss:%.4f eval_acc:%.4f"%(epoch,epochs,t_loss,e_loss,accuracy))
            save_name = "n_gfte_tloss_%.4f_eloss_%.4f_eacc_%.4f.pth"%(t_loss,e_loss,accuracy)
            save_path = os.path.join(store_path,save_name)
            torch.save(model.state_dict(), save_path)


def re_train(t_model,epochs,model_static_path,lr, beta, step_size, gamma,train_data,eval_data,eval_step,stroe_path):
    t_model.load_state_dict(torch.load(model_static_path))
    params = t_model.parameters()
    optimizer, criterion, scheduler = train_init(params, lr, beta, step_size, gamma)
    train_process(t_model, epochs, train_data, eval_data, optimizer, criterion, scheduler, eval_step, stroe_path)

def train(t_model,epochs,lr,beta,step_size,gamma,train_data,eval_data,eval_step,stroe_path):
    t_model.apply(weights_init)
    params = t_model.parameters()
    optimizer,criterion,scheduler = train_init(params,lr,beta,step_size,gamma)
    train_process(t_model,epochs,train_data,eval_data,optimizer,criterion,scheduler,eval_step,stroe_path)


def mian_process(node_feature,out_class,base_channel,train_data_root,eval_data_root,re_train,epochs,lr,beta,step_size,gamma,eval_step,stroe_path,model_static_path,batch_size):
    model = SinglePosNet(node_feature,out_class,base_channel)
    train_data = GFTE_POS_DATASET(train_data_root,None,None)
    eval_data = GFTE_POS_DATASET(eval_data_root,None,None)
    train_d = DataLoader(train_data,batch_size=batch_size,shuffle=True)
    eval_d = DataLoader(eval_data,batch_size=batch_size,shuffle=False)
    if re_train:
        re_train(model,epochs,model_static_path,lr, beta, step_size, gamma,train_data,eval_data,eval_step,stroe_path)
    else:
        train(model,epochs,lr,beta,step_size,gamma,train_d,eval_d,eval_step,stroe_path)




if __name__ == "__main__":
    node_feature = 8
    out_class = 2
    base_channel = 32
    epochs = 80
    train_data_path = r"F:\imgs\SciTSR\mytest"
    eval_data_path = r"F:\imgs\SciTSR\test"
    model_static_path = r""
    store_path = r"F:\python\GFTE\models\pos_model_mg"
    lr = 1e-3
    gamma = 0.9 #学习率下降率
    step_size = 20#每好多次学习率下降
    beta = 0.5
    re_train = False
    eval_step = 5#每好多次eval
    batch_size = 32
    mian_process(node_feature,out_class,base_channel,train_data_path,eval_data_path,re_train,epochs,lr,beta,step_size,gamma,eval_step,store_path,model_static_path,batch_size)
