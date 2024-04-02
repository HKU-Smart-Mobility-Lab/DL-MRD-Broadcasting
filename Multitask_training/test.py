import pickle

train_data = pickle.load(open("../data/train_hk_island_series.pickle","rb"))
test_data = pickle.load(open("../data/test_hk_island_series.pickle","rb"))
print("train len",len(train_data['label']))
print("test len",len(test_data['label']))