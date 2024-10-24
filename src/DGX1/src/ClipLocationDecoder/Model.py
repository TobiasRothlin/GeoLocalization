import torch
from torchinfo import summary

class ClipLocationDecoder(torch.nn.Module):
    def __init__(self,standardization_coordinates=False):
        super(ClipLocationDecoder, self).__init__()
        self.standardization_coordinates = standardization_coordinates

        self.transformer_layer_0 = torch.nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.transformer_layer_1 = torch.nn.TransformerEncoderLayer(d_model=1024, nhead=8)

        self.layer_norm_0 = torch.nn.LayerNorm(1024)
        self.layer_norm_1 = torch.nn.LayerNorm(512)
        self.layer_norm_2 = torch.nn.LayerNorm(256)
        self.layer_norm_3 = torch.nn.LayerNorm(128)


        self.full_connected_0 = torch.nn.Linear(1024, 512)
        self.full_connected_1 = torch.nn.Linear(512, 256)
        self.full_connected_2 = torch.nn.Linear(256, 128)
        self.full_connected_3 = torch.nn.Linear(128, 64)
        self.full_connected_4 = torch.nn.Linear(64, 2)

    def forward(self, x):
        # Of Shape (Batch, 577, 1024)
        x = self.transformer_layer_0(x)
        # Of Shape (Batch, 577, 1024)
        x = self.transformer_layer_1(x)
        # Of Shape (Batch, 577, 1024)
        x = torch.mean(x, dim=1)
        # Of Shape (Batch, 1024)
        x = self.layer_norm_0(x)
        # Of Shape (Batch, 1024)
        x = self.full_connected_0(x)
        x = self.layer_norm_1(x)
        x = torch.nn.functional.tanh(x)
        # Of Shape (Batch, 512)
        x = self.full_connected_1(x)
        x = self.layer_norm_2(x)
        x = torch.nn.functional.tanh(x)
        # Of Shape (Batch, 256)
        x = self.full_connected_2(x)
        x = self.layer_norm_3(x)
        x = torch.nn.functional.tanh(x)
        # Of Shape (Batch, 128)
        x = self.full_connected_3(x)
        x = torch.nn.functional.tanh(x)
        # Of Shape (Batch, 64)
        x = self.full_connected_4(x)

        if self.standardization_coordinates:
            x = torch.nn.functional.tanh(x)
        # Of Shape (Batch, 2)
        return x

    def get_device(self):
        return next(self.parameters()).device
        
        




if __name__ == '__main__':
    input_vector = torch.randn(32, 577, 1024)
    model = ClipLocationDecoder()
    summary(model, input_vector.shape)
    output = model(input_vector)
    print(output.shape)