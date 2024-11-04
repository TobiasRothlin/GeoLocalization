import torch
from torchinfo import summary
from transformers import CLIPModel

class LocationDecoder(torch.nn.Module):

    def __init__(self,config,base_model,use_pre_calculated_embeddings):
        super(LocationDecoder, self).__init__()

        self.use_pre_calculated_embeddings = use_pre_calculated_embeddings
        self.base_model = base_model

        self.use_location_head = config["use_location_head"]
        self.use_similarity_head = config["use_similarity_head"]

        

        if self.use_location_head:
            self.location_head = LocationHeadClip(config["LocationHeadClip"])
        else:
            self.location_head = None

        if self.use_similarity_head:
            self.similarity_head = SimilarityHeadClip(config["SimilarityHeadClip"])
            self.regression_head = None
        else:
            self.similarity_head = None
            self.regression_head = RegressionHead(config["RegressionHead"])

        self.embedding = None

        self.pre_vision_parameters = super(LocationDecoder, self).parameters()

        if self.use_pre_calculated_embeddings:
            self.vision_model = None
        else:
            self.vision_model = ClipVision(self.base_model,return_pooler=not self.use_location_head)

    def get_embedding(self,image):
        self.vision_model = self.vision_model.to(self.get_device())
        if self.use_pre_calculated_embeddings:
            return image
        else:
            self.vision_model.eval()
            with torch.no_grad():
                return self.vision_model(image)
        self.vision_model = self.vision_model.to("cpu")

    def get_parameters(self):
        return self.pre_vision_parameters
    
    def __getstate__(self):
        self.vision_model = None
        state = self.__dict__.copy()
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.use_pre_calculated_embeddings:
            self.vision_model = ClipVision(self.base_model,return_pooler=not self.use_location_head)


        

    def forward(self, x):
        if self.use_location_head:
            x = self.location_head(x)

        self.embedding = x.clone()

        # Shape (Batch, Embedding Dimension)
        if self.use_similarity_head:
            x = self.similarity_head(x)
        else:
            x = self.regression_head(x)
        return x
    
    def get_device(self):
        return next(self.parameters()).device
    
    def summary(self):
        if self.use_pre_calculated_embeddings:
            if self.use_location_head:
                print("Using Pre-Calculated Embeddings and Location Head with input size (1, 577, 1024)")
                summary(self, input_size=(1, 577, 1024),device="cpu")
            else:
                print("Using Pre-Calculated Embeddings and NO Location Head with input size (1, 1024)")
                summary(self, input_size=(1, 1024),device="cpu")
        else:
            print("Using Vision Model and Location Head with input size (1, 3, 336, 336)")
            input_size = (1, 3, 336, 336)
            x = torch.randn(input_size)
            embedding = self.get_embedding(x)
            summary(self, input_size=embedding.shape,device="cpu")

    def save(self, path,show_keys=False):
        state_dict = self.state_dict()
        if show_keys:
            for key in state_dict:
                print(f"Key: {key} Shape: {state_dict[key].shape}")
        torch.save(state_dict, path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        



class ClipVision(torch.nn.Module):

    def __init__(self,base_model,return_pooler=False,freeze_clip_model=False):
        super(ClipVision, self).__init__()
        self.freeze_clip_model = freeze_clip_model
        self.base_model_name = base_model
        self.return_pooler = return_pooler

        self.__setup()

    def __setup(self):
        base_model = CLIPModel.from_pretrained(self.base_model_name)
        self.vision_model = base_model.vision_model

    def forward(self, x):
        if self.return_pooler:
            return self.vision_model(x).pooler_output
        else:
            return self.vision_model(x).last_hidden_state
    

class LocationHeadClip(torch.nn.Module):

    def __init__(self,config):
        super(LocationHeadClip, self).__init__()
        
        self.transformers = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=c["d_model"], nhead=c["nhead"]) for c in config["layers"]])
        self.mean_locatation_head_output = config["mean_locatation_head_output"]
        self.linear_layer_mapping = torch.nn.Linear(in_features=config["linear_layer_mapping"]["in_features"], out_features=config["linear_layer_mapping"]["in_features"], bias=False)


    def forward(self, x):
        for transformer in self.transformers:
            x = transformer(x)

        if self.mean_locatation_head_output:
            x = torch.mean(x, dim=1)
        else:
            x = x[:,0,:]
            x = self.linear_layer_mapping(x)
            x = torch.nn.functional.tanh(x)
        return x
    

class SimilarityHeadClip(torch.nn.Module):

    def __init__(self,config):
        super(SimilarityHeadClip, self).__init__()
        
        layers = []
        for similarityLayerConfig in config["layer_group"]:
            layers.append(SimilarityLayer(similarityLayerConfig))

        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class SimilarityLayer(torch.nn.Module):

    def __init__(self,config):
        super(SimilarityLayer, self).__init__()

        layers = []
        for layer in config:
            if layer["type"] == "Linear":
                layers.append(torch.nn.Linear(in_features=layer["in_features"], out_features=layer["out_features"]))
            elif layer["type"] == "Dropout":
                layers.append(torch.nn.Dropout(p=layer["p"]))
            elif layer["type"] == "LayerNorm":
                layers.append(torch.nn.LayerNorm(normalized_shape=layer["normalized_shape"]))
            elif layer["type"] == "Tanh":
                layers.append(torch.nn.Tanh())
            elif layer["type"] == "ReLU":
                layers.append(torch.nn.ReLU())
            else:
                raise ValueError(f"Layer {layer['type']} not implemented")
        
        self.layer = torch.nn.ModuleList(layers)


    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x
    
class RegressionHead(torch.nn.Module):

    def __init__(self,config):
        super(RegressionHead, self).__init__()

        layers = []
        for regressionLayerConfig in config["layer_group"]:
            layers.append(RegressionLayer(regressionLayerConfig))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class RegressionLayer(torch.nn.Module):
    
    def __init__(self,config):
        super(RegressionLayer, self).__init__()

        layers = []
        for layer in config:
            if layer["type"] == "Linear":
                layers.append(torch.nn.Linear(in_features=layer["in_features"], out_features=layer["out_features"]))
            elif layer["type"] == "Dropout":
                layers.append(torch.nn.Dropout(p=layer["p"]))
            elif layer["type"] == "LayerNorm":
                layers.append(torch.nn.LayerNorm(normalized_shape=layer["normalized_shape"]))
            elif layer["type"] == "Tanh":
                layers.append(torch.nn.Tanh())
            elif layer["type"] == "ReLU":
                layers.append(torch.nn.ReLU())
            else:
                raise ValueError(f"Layer {layer['type']} not implemented")
        
        self.layer = torch.nn.ModuleList(layers)


    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
        return x
    