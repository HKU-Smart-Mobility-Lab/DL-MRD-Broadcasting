import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_class):
        super(MultiMLP, self).__init__()
        self.linear_1 = nn.Linear(input_dim, hidden_dim)
        self.activate = torch.relu
        self.fc1 = nn.Linear(hidden_dim, num_class)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.fc3 = nn.Linear(hidden_dim, num_class)
        self.sigmoid = torch.sigmoid

    def forward(self, inputs):
        inputs = inputs.to(torch.float32)
        y = self.linear_1(inputs)
        y = self.activate(y)
        y1 = self.fc1(y)
        y1 = self.sigmoid(y1)
        y2 = self.fc2(y)
        y2 = self.sigmoid(y2)
        y3 = self.fc3(y)
        y3 = self.sigmoid(y3)
        return torch.cat((y1,y2,y3),1)





class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda")
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to("cuda")

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LogisticModel(torch.nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(LogisticModel, self).__init__()
        self.linear = torch.nn.Linear(input_dimension, output_dimension)

    def forward(self, x):
        y = torch.sigmoid(self.linear(x))
        return y





def test_GridRegressor():
    mlp = GridRegressor(7, 8, 1)
    inputs = torch.rand(8, 7)
    score = mlp(inputs)
    print(score)


if __name__ == "__main__":
    test_GridRegressor()



