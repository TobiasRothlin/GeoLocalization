import torch
import os
import mlflow   
import dotenv

import json

from tqdm import tqdm

from MovingAverage import MovingAverage

from torch.utils.data import DataLoader
from HaversineLoss import HaversineLoss

from torch.utils.tensorboard import SummaryWriter

class SingleGPUTrainer:

    def __init__(self,
                 train_dataset,
                 train_dataloader_config,
                 test_dataset,
                 test_dataloader_config,
                 model,
                 loss_function,
                 optimizer,
                 lr_scheduler,
                 gradient_accumulation_steps,
                 epochs,
                 device,
                 log_interval=10,
                 snapshot_path="./snapshots",
                 log_mlflow=False,
                 mlflow_experiment_name=None,
                 full_run_config=None,
                 contrast_learning_strategy=None,
                 run_name=None):
        
        self.train_dataloader = DataLoader(train_dataset,**train_dataloader_config)
        self.test_dataloader = DataLoader(test_dataset,**test_dataloader_config)
        self.model = model.to(device)
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.log_mlflow = log_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name

        run_name = run_name if run_name is not None else "run"

        self.haversine_loss = HaversineLoss(use_standarized_input=train_dataset.normalize_labels)

        self.contrast_learning_strategy = contrast_learning_strategy

        if self.mlflow_experiment_name is None:
            print("No MLFlow Experiment Name provided. Disabling MLFlow Logging")
            self.log_mlflow = False
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.snapshot_path = snapshot_path
        
        self.loss_average_train = MovingAverage(2*self.gradient_accumulation_steps)
        self.loss_average_test = MovingAverage(2*self.gradient_accumulation_steps)

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        self.snapshot_folder_path = os.path.join(self.snapshot_path, run_name)

        run_idx = 0
        while os.path.exists(self.snapshot_folder_path):
            self.snapshot_folder_path = os.path.join(self.snapshot_path, f"{run_name}_{run_idx}")
            run_idx += 1
        
        os.makedirs(self.snapshot_folder_path)

        if full_run_config is not None:
            with open(os.path.join(self.snapshot_folder_path, "run_config.json"), "w") as f:
                json.dump(full_run_config, f, indent=4)

        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.snapshot_folder_path, "tensorboard"))
        self.training_log_file = os.path.join(self.snapshot_folder_path, "training.log")

        print(f"Model is on Device: {self.model.get_device()}")

        if self.contrast_learning_strategy:
            print("Using Contrast Learning Strategy")
            print(f"Strategy: {self.contrast_learning_strategy}")
            model_input,_,model_output,_ = self.test_dataloader.dataset[0]
        else:
            model_input,model_output = self.test_dataloader.dataset[0]

        print(f"Batch Size: {model_input.shape}")

        model_input = model_input.unsqueeze(0).to(self.device)

        self.tensorboard_writer.add_graph(self.model, model_input)

        del model_input
        del model_output

        print(f"Snapshot folder: {self.snapshot_folder_path}")

        with open(self.training_log_file, "w") as f:
            f.write("")

    def __run_regression(self, batch,batch_idx,is_train):
        input_vec, targets = batch

        

        if is_train:
            input_vec = input_vec.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(input_vec)

            loss = self.loss_function(outputs, targets)

            self.loss_average_train.add(loss.item())

            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                input_vec = input_vec.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(input_vec)

                loss = self.loss_function(outputs, targets)
            self.loss_average_test.add(loss.item())

    def __run_contrast_similarity(self, batch,batch_idx,is_train):
        vec_1, vec_2, location_1, location_2 = batch
        distance = self.haversine_loss.haversine(location_1, location_2)
        distance_norm = distance / 40_000 # Normalize distance to 0,1
        similarity = 1 - distance_norm # Similarity is 1 - distance since the closer the points the more similar they are

        if is_train:
            vec_1 = vec_1.to(self.device)
            vec_2 = vec_2.to(self.device)
            

            embedding_1 = self.model(vec_1)
            embedding_2 = self.model(vec_2)

            if self.contrast_learning_strategy == "CosignSimilarityLoss":
                similarity = similarity.to(self.device)
                loss = self.loss_function(embedding_1, embedding_2, similarity)
            elif self.contrast_learning_strategy == "EuclidianDistanceLoss":
                distance = distance.to(self.device)
                loss = self.loss_function(embedding_1, embedding_2, distance)
            else:
                raise ValueError("Contrast Learning Strategy not recognized")

            self.loss_average_train.add(loss.item())

            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            with torch.no_grad():
                vec_1 = vec_1.to(self.device)
                vec_2 = vec_2.to(self.device)
                distance = distance.to(self.device)

                embedding_1 = self.model(vec_1)
                embedding_2 = self.model(vec_2)

                loss = self.loss_function(embedding_1, embedding_2, distance)
            self.loss_average_test.add(loss.item())




    def __run_single_batch(self, batch,batch_idx,is_train):
        if self.contrast_learning_strategy:
            self.__run_contrast_similarity(batch,batch_idx,is_train)
        else:
            self.__run_regression(batch,batch_idx,is_train)
    
    def __run_epoch(self, dataloader, epoch,with_logging=True,is_train=True):
        prefix = "Train" if is_train else "Test"
        progress_bar = tqdm(dataloader, desc=f"{prefix}Epoch {epoch}", postfix=f"Loss {self.loss_average_train.get():.5f}")
        for batch_idx, batch in enumerate(progress_bar):
            self.__run_single_batch(batch,batch_idx,is_train)

            if is_train:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_train.get():.5f}")
                self.log(f"Train,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_train.get():.5f}")
                self.tensorboard_writer.add_scalar('Loss/train', self.loss_average_train.get(), epoch * len(dataloader) + batch_idx)
            else:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_test.get():.5f}")
                self.log(f"Test,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_test.get():.5f}")
                self.tensorboard_writer.add_scalar('Loss/test', self.loss_average_test.get(), epoch * len(dataloader) + batch_idx)

            if with_logging:
                if batch_idx % self.log_interval == 0:
                    self.save(os.path.join(self.snapshot_folder_path, f"epoch_{epoch}_batch_{batch_idx}.pt"))
                    

        progress_bar.close()

        try:
            if is_train:
                if self.log_mlflow:
                    mlflow.log_metric("train_loss", self.loss_average_train.get(), step=epoch)
                    
            else:
                if self.log_mlflow:
                    mlflow.log_metric("test_loss", self.loss_average_test.get(), step=epoch)
        except Exception as e:
            print("Could not connect to MLFlow")
            print(e)
            self.log_mlflow = False
    
    

    def train(self,epochs = None):
        if epochs is not None:
            self.epochs = epochs

        if self.log_mlflow:
            try:
                dotenv.load_dotenv(dotenv.find_dotenv())
                os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
                os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
                mlflow.set_tracking_uri("https://mlflow.infs.ch")
                mlflow.set_experiment(self.mlflow_experiment_name)
                mlflow.start_run()
            except Exception as e:
                print("Could not connect to MLFlow")
                print(e)
                self.log_mlflow = False

        for epoch in range(self.epochs):
            self.model.train()
            self.__run_epoch(self.train_dataloader, epoch,with_logging=True,is_train=True)

            self.model.eval()
            self.__run_epoch(self.test_dataloader, epoch,with_logging=False,is_train=False)

            self.lr_scheduler.step()

            self.save(os.path.join(self.snapshot_folder_path, f"model_end_of_epoch_{epoch}.pt"))

        if self.log_mlflow:
            try:
                self.save(os.path.join(self.snapshot_folder_path, "model_final.pt"))
                mlflow.log_artifact(os.path.join(self.snapshot_folder_path, "model_final.pt"))
                mlflow.end_run()
            except Exception as e:
                print("Could not connect to MLFlow")
                print(e)
                self.log_mlflow = False

        self.tensorboard_writer.close()


    def save(self, path):
        self.model.save(path)


    def log(self, message):
        with open(self.training_log_file, "a") as f:
            f.write(message)
            f.write("\n")
        
    