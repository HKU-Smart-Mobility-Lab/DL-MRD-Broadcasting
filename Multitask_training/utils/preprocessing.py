  # -*- coding: utf-8 -*

import numpy as np
import pandas  as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys

# This function is used to generate formulated data for time series model
# The data for training and testing will be stored in data file in the parent directory
# You just need to call this function for one time
def prepare_series_data(data,scaler, seq_len, stride,train_ratio,train_data_path,test_data_path):
    
    grid_id_list = data['grid_id'].unique()
    pred_len = 1  # 预测长度
    X, Y = [], []  # 用来存放特征和标签
    data_norm = scaler.fit_transform(data)
    data = pd.DataFrame(data_norm,columns=data.columns)
    print(data)
    for grid_id in grid_id_list:
        # 取某个grid的数据
        grid_data = data[data['grid_id'] == grid_id]
        print("processing grid:", grid_id)
        # 将该grid的数据进行排序
        grid_data = grid_data.sort_values("time_stamp")
        # 切分时间片
        for i in range(0, len(grid_data) - seq_len - pred_len, stride):
            X.append(grid_data.iloc[i:i + seq_len, :-6].values)
            Y.append(grid_data.iloc[i + seq_len + pred_len, -6:].values)

    train_X, test_X, train_Y, test_Y = train_test_split(
        np.array(X), np.array(Y), train_size=train_ratio, random_state=42
    )
    train_data = {"data":train_X,'label':train_Y}
    test_data = {"data":test_X,'label':test_Y}
    pickle.dump(train_data,open(train_data_path,"wb"))
    pickle.dump(test_data,open(test_data_path,"wb"))

    

if __name__ == "__main__":
    data_path = "../../data/HongKong_whole.csv"
    data = pd.read_csv(data_path)
    # columns are time_stamp,time_period,grid_id,num_order,num_matched_order,num_available_driver,avg_matched_pickup_distance,avg_matched_price,avg_pickup_distance,avg_price,radius,driver_utilization_rate,total_matched_price
    new_colume_order = ['time_stamp','time_period','grid_id','num_available_driver','avg_pickup_distance','avg_price','radius','num_order','num_matched_order','avg_matched_pickup_distance','avg_matched_price','driver_utilization_rate','total_matched_price','matched_ratio']
    data['matched_ratio'] = 0
    data['matched_ratio'] = data.apply(lambda row: row['num_matched_order'] / row['num_order'] if row['num_order']!=0 else 0,axis=1)
    data = data[new_colume_order]
    scaler = MinMaxScaler()
    prepare_series_data(data,scaler,100, 1,0.7,"../../data/train_HK_whole_series.pickle", "../../data/test_HK_whole_series.pickle")
    
   