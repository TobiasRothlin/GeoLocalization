from HaversineLoss import HaversineLoss

import torch

from tqdm import tqdm

class Evaluation:
    def __init__(self, model, data, device,user_standardized_input=False):
        self.model = model
        self.data_loader = data
        self.device = device
        self.haversine_loss = HaversineLoss(use_standarized_input=user_standardized_input)

        self.model = self.model.to(self.device)

        self.init_evaluation_results()

    def init_evaluation_results(self):
        self.evaluation_results = {
            "average_loss": 0,
            "is_inside":
            {
                1: 0,
                25: 0,
                200: 0,
                750: 0,
                2500: 0,
            },
            "total": 0,
            "batches": 0
        }

    def eval_single_batch(self, images, labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        outputs = self.model(images)

        distances = self.haversine_loss.haversine(outputs, labels)
        loss = self.haversine_loss(outputs, labels).item()
        return loss, distances
    

    def evaluate(self):
        self.init_evaluation_results()
        progress_bar = tqdm(enumerate(self.data_loader), desc="Evaluating", total=len(self.data_loader))

        self.model.eval()
        with torch.no_grad():
            for i, (images, labels) in progress_bar:
                loss, distances = self.eval_single_batch(images, labels)
                self.evaluation_results["average_loss"] += loss
                self.evaluation_results["total"] += labels.size(0)
                self.evaluation_results["batches"] += 1

                for distance in distances:
                    for key in self.evaluation_results["is_inside"]:
                        if distance <= key:
                            self.evaluation_results["is_inside"][key] += 1

                progress_bar.set_postfix_str(f"Loss: {self.evaluation_results['average_loss']/self.evaluation_results['batches']:.5f}")

        progress_bar.close()
        self.evaluation_results["average_loss"] /= len(self.data_loader)

        self.evaluation_results["is_inside_average"] = {key: value / self.evaluation_results["total"] for key, value in self.evaluation_results["is_inside"].items()}     
        return self.evaluation_results
    
    def __str__(self):
        output = ""
        output += "Average Loss: " + str(self.evaluation_results["average_loss"]) + "\n"
        output += "Accuracy by distance:\n"
        for key in self.evaluation_results["is_inside"]:
            output += f"  -{key} km: {self.evaluation_results['is_inside'][key]}/{self.evaluation_results['total']} ({self.evaluation_results['is_inside_average'][key]*100:.2f}%)\n"

        return output

        
       

