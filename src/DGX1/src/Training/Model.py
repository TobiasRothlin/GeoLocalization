from transformers import CLIPProcessor, CLIPModel
from torch.nn import Module

class GeoLocalizationModel(Module):

    def __init__(self, base_model_name,device):
        super(GeoLocalizationModel, self).__init__()

        self.device = device

        self.base_model = CLIPModel.from_pretrained(base_model_name)

        self.vision_model = self.base_model.vision_model

        self.vision_model.to(device)




        
    def forward(self, x):
        x = self.vision_model(x)
        return x.last_hidden_state