import torch.nn as nn
import torch


# class MyLoss(nn.Module):
#     def __init__(self,id,k):
#         super(MyLoss,self).__init__()
#         self.id = id
#         self.k = k
    

#     def forward(self,x,y):
#         return torch.mean(self.k * torch.pow(x[:,self.id]-y[:,self.id],2))

class MyLoss(nn.Module):
    def __init__(self,obj_num,k):
        super(MyLoss,self).__init__()
        self.obj_num = obj_num
        self.k = k

    def update_k(self,k):
        self.k = k
    

    def forward(self,x,y):
        # print(x[:,5])

        # print(y[:,5])
        loss = torch.tensor(0,dtype=torch.float32,requires_grad=True)
        step_losses = []
        for i in range(self.obj_num):
            # print(x.size(),y.size())
            step_loss =  torch.mean(torch.pow(x[:,i]-y[:,i],2))
            loss = loss + self.k[i] * step_loss
            step_losses.append(step_loss)

        return loss, torch.tensor(step_losses).detach().numpy()
def testloss():
    myloss = MyLoss(1,1,1)
    print(myloss(torch.tensor([[2,2,2],[2,2,2]]),torch.tensor([[2.1,2.1,2.1],[2.1,2.1,2.1]])))

if __name__ == "__main__":
    testloss()