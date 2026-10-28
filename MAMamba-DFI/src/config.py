import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)
 
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]

        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


DEFAULT_CONFIG = {
    'MODE': 1,                      
    'MODEL': 1,                     
    'MASK': 3,                      
    'NMS': 1,                       
    'SEED': 10,                     
    'GPU': [0],                     
    'AUGMENTATION_TRAIN': 0,        

    'LR': 0.0001,                   
    'D2G_LR': 0.1,                  
    'BETA1': 0.0,                   
    'BETA2': 0.9,                   
    'BATCH_SIZE': 4,                
    'INPUT_SIZE': 256,              
    'MAX_ITERS': 2e6,               

    'L1_LOSS_WEIGHT': 1,            
    'STYLE_LOSS_WEIGHT': 1,         
    'CONTENT_LOSS_WEIGHT': 1,      
    'INPAINT_ADV_LOSS_WEIGHT': 0.01,
    'FFL_LOSS_WEIGHT': 10.0,
    'TV_LOSS_WEIGHT': 0.1,          

    'GAN_LOSS': 'lsgan',            
    'GAN_POOL_SIZE': 0,            

    'SAVE_INTERVAL': 1000,          
    'SAMPLE_INTERVAL': 1000,        
    'SAMPLE_SIZE': 12,              
    'EVAL_INTERVAL': 0,             
    'LOG_INTERVAL': 10,             
}
