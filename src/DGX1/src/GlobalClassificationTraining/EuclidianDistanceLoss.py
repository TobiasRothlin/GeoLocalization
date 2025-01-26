import torch

class EuclidianDistanceLoss(torch.nn.Module):
    def __init__(self):
        super(EuclidianDistanceLoss, self).__init__()

    def euclidian_distance(self, vec_1, vec_2):
        return torch.sum((vec_1 - vec_2)**2, dim=1)

    def forward(self, vec_1, vec_2,distance):
        similarity = self.euclidian_distance(vec_1, vec_2)
        loss = torch.nn.functional.mse_loss(similarity, distance)
        return loss



        