import torch

class HaversineLoss(torch.nn.Module):
    def __init__(self,use_standarized_input=False):
        super(HaversineLoss, self).__init__()
        self.earth_radius = 6371.0
        self.pi = torch.acos(torch.zeros(1)).item() * 2

        self.use_standarized_input = use_standarized_input

        if self.use_standarized_input:
            print("Using Standarized Input")

    def haversine(self, pred_location, target_location):
        pred_lat, pred_lon = pred_location[:, 0], pred_location[:, 1]
        target_lat, target_lon = target_location[:, 0], target_location[:, 1]

        if self.use_standarized_input:
            pred_lat = pred_lat * 90
            pred_lon = pred_lon * 180
            target_lat = target_lat * 90
            target_lon = target_lon * 180

        delta_lat = self.__radians(pred_lat - target_lat)

        alpha_0 = torch.pow(torch.sin(delta_lat / 2), 2)
        alpha_1 = torch.cos(self.__radians(target_lat)) * torch.cos(self.__radians(pred_lat))
        alpha_2 = torch.pow(torch.sin(self.__radians(pred_lon - target_lon) / 2), 2)

        haversign = 2 * self.earth_radius * torch.asin(torch.sqrt(alpha_0 + alpha_1 * alpha_2))
        return haversign

    def forward(self, pred_location, target_location):
        haversign = self.haversine(pred_location, target_location)
        return torch.mean(haversign)


    def __radians(self, x):
        return x * (self.pi / 180)



if __name__ == '__main__':

    loss_funciton = HaversineLoss()

    pred = torch.tensor([[47.217209, 8.820637],[47.217209, 8.820637]])
    target = torch.tensor([[26.129752, -34.360865],[26.129752, -34.360865]])

    dist = loss_funciton(pred, target)
    print(dist)








