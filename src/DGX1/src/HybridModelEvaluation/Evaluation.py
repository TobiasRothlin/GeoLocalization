import torch

from tqdm import tqdm

class Evaluation:
    def __init__(self, classification_model,idx_to_labels,regression_models, data, device,loss_function):
        self.classification_model = classification_model
        self.regression_models = regression_models
        self.idx_to_labels = idx_to_labels
        self.data_loader = data
        self.device = device
        self.haversine_loss = loss_function

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
        self.classification_model = self.classification_model.to(self.device)
        outputs = self.classification_model(images)
        argmax = torch.argmax(outputs, dim=1)
        predicted_continents = [self.idx_to_labels[idx.item()] for idx in argmax]

        self.classification_model = self.classification_model.to("cpu")
        predicted_locations = []
        index = 0
        for predicted_continent in predicted_continents:
            model = self.regression_models[predicted_continent].to(self.device)
            image = images[index].unsqueeze(0)
        
            predicted_locations.append(model(image).to("cpu"))
            model = model.to("cpu")
            index += 1
        
        outputs = torch.cat(predicted_locations, dim=0)

        outputs = outputs.to(self.device)


        distances = self.haversine_loss.haversine(outputs, labels)
        loss = self.haversine_loss(outputs, labels).item()
        return loss, distances
    

    def evaluate(self):
        self.init_evaluation_results()
        progress_bar = tqdm(enumerate(self.data_loader), desc="Evaluating", total=len(self.data_loader))

        self.classification_model.eval()
        for model_name in self.regression_models:
            self.regression_models[model_name].eval()
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
    
    def to_file(self, file_path,name=None):
        if not file_path.endswith(".txt"):
            file_path += ".txt"
        with open(file_path, "w") as file:
            file.write("Evaluation Results\n")
            file.write("====================================\n")
            if name:
                file.write(name+"\n")

            file.write(str(self))

        
       

