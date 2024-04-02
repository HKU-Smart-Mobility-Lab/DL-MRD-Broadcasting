import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import config
import pickle


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, stand_scaler, model='train', regoressor='lr', seq_len=5, stride=1):

        if regoressor == 'lstm_regressor' or regoressor == 'transformer':
            data = pickle.load(open(filepath,"rb"))
            # scalar = MinMaxScaler()
            # normalized_data = pd.DataFrame(scalar.fit_transform(data.to_numpy()),columns=data.colmuns)
            self.x_data,self.y_data =data['data'],data['label'][:,:6]
            # print(self.x_data.shape)
            # self.x_data = scalar.fit_transform(self.x_data)
            # self.y_data = scalar.fit_transform(self.y_data)
            # print(self.y_data.shape)
        # z = np.loadtxt(filepath,dtype=np.float32,delimiter = ",")
        else:
            z = pd.read_csv(filepath)
            z_arr = z.to_numpy()
            feature_data = z_arr[:,0:-6]
            feature_scaler = MinMaxScaler()
            feature_scaler.fit(feature_data)
            pickle.dump(feature_scaler,open('8_feature_scalar_manhattan.pickle','wb'))
            if model == 'train':
                stand_scaler.fit(z)
                z = stand_scaler.transform(z)
            else:
                z = stand_scaler.transform(z)


            self.x_data = z[:, 0:-6]
            # print(self.x_data[:,5])
            self.y_data = z[:, -5:-1]


        self.length = len(self.x_data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def create_sequences(self, x_data,y_data, seq_length, stride):
        x,y = [],[]
        for i in range(0,len(x_data) - seq_length - 1,stride):
            seq = x_data[i:i + seq_length]
            x.append(seq)
            y.append(y_data[i + seq_length + 1])
        return np.array(x),np.array(y)

    def date_preprocessing(self):

        contain_nan_x = (True in np.isnan(self.x_data))
        contain_nan_y = (True in np.isnan(self.y_data))
        print(contain_nan_x, contain_nan_y)
        return self.x_data, self.y_data
    

