from model.GridRegressor import MultiMLP,LogisticModel,LSTMNetwork
from model.transformer.Transformer import Transformer
class ModelGetter():
    def __init__(self,config):
        super(ModelGetter,self).__init__()
        self.config = config

    def get_model(self):
        if self.config['model'] == "multimlp":
            model = MultiMLP(self.config['input_dim'],self.config['hidden_dim'],self.config['num_class'])
        elif self.config['model'] == "logistic_regressor":
            model = LogisticModel(self.config['input_dim'],self.config['num_class'])
        elif self.config['model'] == "lstm_regressor":
            model = LSTMNetwork(self.config['input_dim'],self.config['hidden_dim'], self.config['num_layers'],self.config['num_class'])
        elif self.config['model'] == "transformer":
            model = Transformer()
        return model