from transformers import CLIPProcessor, CLIPModel
from torch.nn import Module
from torchsummary import summary
import torch

class GeoLocalizationRegressionHead(Module):
    
        def __init__(self):
            super(GeoLocalizationRegressionHead, self).__init__()
    
            self.ff_layers = torch.nn.ModuleList([
                torch.nn.Linear(1024, 512),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(512),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(),
                torch.nn.LayerNorm(256),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 2),
                torch.nn.Tanh()
            ])
    
            self.apply(self._init_weights)
            
    
        def _init_weights(self, m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        def forward(self, x):
            for layer in self.ff_layers:
                x = layer(x)
            return x

        def to(self, *args, **kwargs):
            return super().to(*args, **kwargs)

class GeoLocalizationModel(Module):

    def __init__(self, base_model_name):
        super(GeoLocalizationModel, self).__init__()

        base_model = CLIPModel.from_pretrained(base_model_name)

        self.vision_model = base_model.vision_model
        self.reg_head = GeoLocalizationRegressionHead()
        


    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.vision_model = self.vision_model.to(*args, **kwargs)
        self.reg_head = self.reg_head.to(*args, **kwargs)
        return self
        
    def forward(self, x):
        x = self.vision_model(x).last_hidden_state
        x = x[:, 0, :]
        x = self.reg_head(x)
        return x
    
    def load(self, path):
        self.load_state_dict(torch.load(path))