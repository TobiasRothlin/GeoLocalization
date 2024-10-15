import torch

class HaversineLoss(torch.nn.Module):
    def __init__(self):
        super(HaversineLoss, self).__init__()
        self.earth_radius = 6371.0
        self.pi = torch.acos(torch.zeros(1)).item() * 2

    def forward(self, pred_location, target_location):
        pred_lat, pred_lon = pred_location[:, 0], pred_location[:, 1]
        target_lat, target_lon = target_location[:, 0], target_location[:, 1]

        delta_lat = self.__radians(pred_lat - target_lat)

        alpha_0 = torch.pow(torch.sin(delta_lat / 2), 2)
        alpha_1 = torch.cos(self.__radians(target_lat)) * torch.cos(self.__radians(pred_lat))
        alpha_2 = torch.pow(torch.sin(self.__radians(pred_lon - target_lon) / 2), 2)

        haversign = 2 * self.earth_radius * torch.asin(torch.sqrt(alpha_0 + alpha_1 * alpha_2))

        return haversign

    def __radians(self, x):
        return x * (self.pi / 180)



if __name__ == '__main__':
    from haversine import haversine, Unit

    loss = HaversineLoss()
    pred = torch.tensor([[45.7597, 4.8422], [48.8566, 2.3522]])
    target = torch.tensor([[48.8567, 2.3508], [48.8566, 2.3522]])
    print(loss(pred, target))

    print(haversine((45.7597, 4.8422), (48.8567, 2.3508), unit=Unit.KILOMETERS))
