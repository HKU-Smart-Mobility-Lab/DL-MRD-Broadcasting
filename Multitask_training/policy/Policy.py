import numpy as np

class AMPolicy:
    def __init__(self,config):
        print("AM Policy initialized...")
        self.object_num = config['object_num']
        self.init_interval_losses(self.object_num)
        self.update_time = 1

    def init_interval_losses(self,obj_num):
        self.interval_losses = np.zeros(obj_num)

    def update_losses(self,step_losses,train_step):
        self.interval_losses = (self.interval_losses *(train_step - 1) +step_losses) / train_step

    def update_config(self,config):

        max_idx = np.argmax(self.interval_losses)
        config['k'] = [0] * self.object_num
        config['k'][max_idx] = 1

        # 如果不保留全部历史loss，则每policy_update_step哥step就将loss清零
        if not config['total_history']:
            self.init_interval_losses(self.object_num)

        self.update_time += 1
        return config
    

class AMWeightedPolicy:
    def __init__(self,config):
        print("AM Policy initialized...")
        self.object_num = config['object_num']
        self.init_interval_losses(self.object_num )
        self.update_time = 1

    def init_interval_losses(self,obj_num):
        self.interval_losses = np.zeros(obj_num)

    def update_losses(self,step_losses,train_step):
        self.interval_losses = (self.interval_losses * (train_step - 1) + step_losses)  / train_step

    def update_config(self,config):

        config['k'] = self.interval_losses * 4.0 / sum(self.interval_losses)

        # 如果不保留全部历史loss，则每policy_update_step哥step就将loss清零
        if not config['total_history']:
            self.init_interval_losses(self.object_num)

        self.update_time += 1
        return config


class ExSmWeightedPolicy:
    def __init__(self,config):
        print("Exponential Smoothing Policy initialized...")
        self.object_num = config['object_num']
        self.init_interval_losses(self.object_num)
        self.omega = config['omega']
        self.update_time = 1

    def init_interval_losses(self,obj_num):
        self.interval_losses = np.zeros(obj_num)

    def update_losses(self,step_losses,train_step):
        self.interval_losses = (self.interval_losses * self.omega *(train_step - 1) + step_losses) / train_step

    def update_config(self,config):
        config['k'] = self.interval_losses * 1.0 / sum(self.interval_losses)


        self.update_time += 1
        # 如果不保留全部历史loss，则每policy_update_step哥step就将loss清零
        if not config['total_history']:
            self.init_interval_losses(self.object_num)
        
        return config


class ExSmPolicy:
    def __init__(self,config):
        print("Exponential Smoothing Policy initialized...")
        self.object_num = config['object_num']
        self.init_interval_losses(self.object_num)
        self.omega = config['omega']
        self.update_time = 1
        
    def init_interval_losses(self,obj_num):
        self.interval_losses = np.zeros(obj_num)


    def update_losses(self,step_losses,train_step):
        self.interval_losses = (self.interval_losses * self.omega * (train_step - 1)  + step_losses) / train_step

    def update_config(self,config):

        max_idx = np.argmax(self.interval_losses)
        config['k'] = [0] * self.object_num
        config['k'][max_idx] = 1
        self.update_time += 1
        # 如果不保留全部历史loss，则每policy_update_step哥step就将loss清零
        if not config['total_history']:
            self.init_interval_losses(self.object_num)
        return config



class FixedPolicy:
    def __init__(self,config):
        print("FixedPolicy initialized...")

    def update_config(self,config):
        return config

    def update_losses(self,step_losses,train_step):
        return 
    
    
        

class PolicyGetter:
    def __init__(self,config):
        self.policy = config['policy']
        self.config = config
    
    def update_policy(self,policy):
        self.policy = policy
    
    def get_policy(self):
        if self.policy == "fixed":
            return FixedPolicy(self.config)
        elif self.policy == "am":
            return AMPolicy(self.config)
        elif self.policy == "es":
            return ExSmPolicy(self.config)
        elif self.policy == "amw":
            return AMWeightedPolicy(self.config)
        elif self.policy == "esw":
            return ExSmWeightedPolicy(self.config)

if __name__ == "__main__":
    pass