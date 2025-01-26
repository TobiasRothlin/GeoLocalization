import torch

class CosignSimilarityLoss(torch.nn.Module):
    def __init__(self):
        super(CosignSimilarityLoss, self).__init__()

    def forward(self, vec_1, vec_2,similarity):
        similarity = similarity * 2 - 1 # Normalize similarity to -1,1 from 0,1
        output = torch.cosine_similarity(vec_1, vec_2)
        loss = torch.nn.functional.mse_loss(output, similarity)
        return loss



        