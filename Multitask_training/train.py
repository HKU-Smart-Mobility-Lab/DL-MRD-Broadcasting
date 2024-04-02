import yaml
import torch
from utils.ModelGetter import ModelGetter
from tensorboardX import SummaryWriter
from data_process.MyDataset import MyDataset
from torch.utils.data import DataLoader
from loss.MyLoss import MyLoss
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import time
import os
from tqdm import tqdm
from policy.Policy import PolicyGetter
import numpy as np
import json
# load config
CONFIG_FILE = "config/config.yaml"
print(torch.cuda.is_available())

with open(CONFIG_FILE,"r",encoding="utf-8") as f:
    config = yaml.safe_load(f)


cur_time = time.asctime( time.localtime(time.time()) )

cur_time = cur_time.replace(':', '_')
writer = SummaryWriter("runs/" + config['model'] + "_" + config['policy'] + "/" + cur_time  + "/")
os.makedirs("ckpt/model_{}_policy_{}/{}".format(config['model'],config['policy'],cur_time))
with open("runs/" + config['model'] + "_" + config['policy'] + "/" + cur_time  + "/" +"config.txt",'w') as f:
    f.write(json.dumps(config))
    
scaler = MinMaxScaler() #StandardScaler()
# load dataset
train_dataset = MyDataset(config['train_data_path'],scaler,"train",config['model'])
train_dataloader = DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=True,num_workers=3)
test_dataset = MyDataset(config['test_data_path'],scaler,"valid",config['model'])
test_dataloader = DataLoader(test_dataset,batch_size=config['batch_size'],shuffle=False,num_workers=3)


train_data_size = len(train_dataloader)
test_data_size = len(test_dataloader)

# load model
model_getter = ModelGetter(config)
model = model_getter.get_model()
device = config['device']
model = model.to(device)
# load policy
policy_getter = PolicyGetter(config)
policy = policy_getter.get_policy()

# load criterion and optimizer
# criterion = torch.nn.MSELoss(size_average=None)#
# criterion = MyLoss(config['object_num'],config['k'])

# if config['model'] == 'grid_regressor':
    # criterion = []
    # for i in range(config['object_num']):
    #     loss = MyLoss(i,config['k'][i]) 
    #     criterion.append(loss)w
criterion = MyLoss(config['object_num'],config['k'])

    # criterion = torch.nn.MSELoss(size_average=None)

optim = torch.optim.SGD(model.parameters(),lr=1e-5)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0)



if config['save_best']:
    best_loss = float("inf")

# training process
for epoch in tqdm(range(config['epoch'])):
    print("-------epoch  {} -------".format(epoch+1))
    total_train_loss = 0
    # total_train_weighted_loss = 0
    model.train()
    for i,[x,y] in enumerate(train_dataloader):
        # print("label",y)
        x = x.to(torch.float32).to(device)
        # print("Input data:",x)
        y = y.to(torch.float32).to(device)
        y_ = model(x)
        # print("Predict result:",y_)
        optim.zero_grad()

        train_step = len(train_dataloader)*epoch+i+1
        # if config['model'] == "grid_regressor":
            ## 进行策略更新
            # step_losses = [criterion[idx](y_,y) for idx in range(config['object_num'])]
            # step_losses_ = torch.tensor(step_losses).detach().numpy()
            # total_train_loss += np.sum(step_losses_)
            # step_losses_weighted = step_losses_ * np.array(config['k'])
            # total_train_weighted_loss += np.sum(step_losses_weighted)
            # total_loss = torch.tensor(0,dtype = torch.float32,requires_grad=True)
            # for idx,(loss,k) in enumerate(zip(step_losses,config['k'])):
            #     total_loss = total_loss + k*loss
            #     writer.add_scalar("train_loss_{}".format(idx), loss.item(), i)
            
            # total_loss.backward(retain_graph=True)
        loss,step_losses = criterion(y_,y)
        loss.backward()
        total_train_loss += loss.item()
        print("step losses",step_losses)
        writer.add_scalars("train loss for each obj",{str(i):x for i,x in enumerate(step_losses)}, train_step)
        policy.update_losses(np.array(step_losses),train_step)
        if (i + 1) % config['policy_update_step'] == 0:
            config = policy.update_config(config)
            criterion.update_k(config['k'])
            writer.add_scalars("value of k",{str(i):x for i,x in enumerate(config['k'])}, train_step)
                
        # else:
        #     loss = criterion(y_,y)
        #     loss.backward()
        #     total_train_loss += loss.item()

        optim.step()
        scheduler.step()

        # if train_step % 100 == 0:
        #     print("train step:{}, Loss: {}".format(train_step, loss.item()))
    print("train epoch:{}, Loss: {}".format(epoch+1, total_train_loss / train_data_size))
    writer.add_scalar("train_loss", total_train_loss / train_data_size, epoch)
    # writer.add_scalar("train_weighted_loss", total_train_weighted_loss, epoch)
    
    

    if (epoch + 1) % config['eval_epoch'] == 0:
        model.eval()
        total_test_loss = 0
        # total_test_weighted_loss = 0
        with torch.no_grad():
            for i,[x, y] in enumerate(test_dataloader): 
                x = x.to(torch.float32).to(device)
                y = y.to(torch.float32).to(device)

                y_ = model(x)
                test_step = len(test_dataloader)*epoch+i+1
                # if config['model'] == "grid_regressor":
                    # step_losses = [criterion[idx](y_,y) for idx in range(config['object_num'])]
                    # step_losses_ = torch.tensor(step_losses).detach().numpy()
                    # total_test_loss += np.sum(step_losses_)
                    # step_losses_ = step_losses_ * np.array(config['k'])
                    # total_test_weighted_loss += np.sum(step_losses_)

                    # for idx,(loss,k) in enumerate(zip(step_losses,config['k'])):
                    #     writer.add_scalar("test_loss_{}".format(idx), loss.item(), i)
                loss,step_losses = criterion(y_,y)
                writer.add_scalars("test loss for each obj",{str(i):x for i,x in enumerate(step_losses)}, test_step)
                total_test_loss = total_test_loss + loss.item()
                
                # else:
                #     loss = criterion(y_,y)
                #     total_test_loss = total_test_loss + loss.item()
    
        print("test set Loss: {}".format(total_test_loss / test_data_size))
        writer.add_scalar("test_total_loss", total_test_loss / test_data_size, epoch)

        # writer.add_scalar("test_total_weighted_loss", total_test_weighted_loss, epoch)
        

        if config['save_best'] == True:
            if total_test_loss < best_loss:
                best_loss = total_test_loss
                torch.save(model, "ckpt/model_{}_policy_{}/{}/best_model.pth".format(config['model'],config['policy'],cur_time))


    


    if (epoch + 1) % config['save_epoch'] == 0:
        torch.save(model, "ckpt/model_{}_policy_{}/{}/epoch={}.pth".format(config['model'],config['policy'],cur_time,epoch+1))

    





