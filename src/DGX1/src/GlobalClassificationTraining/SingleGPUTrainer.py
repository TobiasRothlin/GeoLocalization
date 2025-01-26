import torch
import os
import mlflow   
import dotenv

import shutil

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
                 run_name=None,
                 output_lable_dict_train=None,
                 output_lable_dict_test=None):
        
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

        if self.mlflow_experiment_name is None:
            print("No MLFlow Experiment Name provided. Disabling MLFlow Logging")
            self.log_mlflow = False
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.snapshot_path = snapshot_path
        
        self.loss_average_train = MovingAverage(2*self.gradient_accumulation_steps)
        self.accuracy_average_train = MovingAverage(2*self.gradient_accumulation_steps)
        self.loss_average_test = MovingAverage(2*self.gradient_accumulation_steps)
        self.accuracy_average_test = MovingAverage(2*self.gradient_accumulation_steps)

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

        if output_lable_dict_train is not None:
            shutil.copy(output_lable_dict_train, os.path.join(self.snapshot_folder_path, "output_lable_dict_train.json"))

        if output_lable_dict_test is not None:
            shutil.copy(output_lable_dict_test, os.path.join(self.snapshot_folder_path, "output_lable_dict_test.json"))

        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.snapshot_folder_path, "tensorboard"))
        self.training_log_file = os.path.join(self.snapshot_folder_path, "training.log")

        print(f"Model is on Device: {self.model.get_device()}")

        model_input,model_output = self.test_dataloader.dataset[0]

        print(f"Batch Size: {model_input.shape}")
        print(f"Model Output: {model_output.shape}")

        model_input = model_input.unsqueeze(0).to(self.device)

        self.tensorboard_writer.add_graph(self.model, model_input)

        del model_input
        del model_output

        print(f"Snapshot folder: {self.snapshot_folder_path}")

        with open(self.training_log_file, "w") as f:
            f.write("")

    def evaluate_accuracy_on_batch(self,output, targets):
        output = torch.clone(output.detach().cpu())
        targets = torch.clone(targets.detach().cpu())

        _, predicted = torch.max(output, 1)
        _, true_labels = torch.max(targets, 1)
        correct = (predicted == true_labels).sum().item()
        total = targets.size(0)
        return correct / total
        


    def __run_classification(self, batch,batch_idx,is_train):
        input_vec, targets = batch

        if is_train:
            input_vec = input_vec.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(input_vec)

            acc = self.evaluate_accuracy_on_batch(outputs, targets)

            loss = self.loss_function(outputs, targets)

            self.loss_average_train.add(loss.item())
            self.accuracy_average_train.add(acc)

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
                acc = self.evaluate_accuracy_on_batch(outputs, targets)
            self.accuracy_average_test.add(acc)
            self.loss_average_test.add(loss.item())





    def __run_single_batch(self, batch,batch_idx,is_train):
        self.__run_classification(batch,batch_idx,is_train)
    
    def __run_epoch(self, dataloader, epoch,with_logging=True,is_train=True):
        prefix = "Train" if is_train else "Test"
        progress_bar = tqdm(dataloader, desc=f"{prefix}Epoch {epoch}", postfix=f"Loss:{self.loss_average_train.get():.5f} Acc:{self.accuracy_average_train.get()*100:.5f}%")
        for batch_idx, batch in enumerate(progress_bar):
            self.__run_single_batch(batch,batch_idx,is_train)

            if is_train:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_train.get():.5f} Acc:{self.accuracy_average_train.get()*100:.5f}%")
                self.log(f"Train,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_train.get():.5f},Acc:{self.accuracy_average_train.get()*100:.5f}%")
                self.tensorboard_writer.add_scalar('Loss/train', self.loss_average_train.get(), epoch * len(dataloader) + batch_idx)
                self.tensorboard_writer.add_scalar('Accuracy/train', self.accuracy_average_train.get(), epoch * len(dataloader) + batch_idx)
            else:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_test.get():.5f} Acc:{self.accuracy_average_train.get()*100:.5f}%")
                self.log(f"Test,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_test.get():.5f},Acc:{self.accuracy_average_test.get()*100:.5f}%")
                self.tensorboard_writer.add_scalar('Loss/test', self.loss_average_test.get(), epoch * len(dataloader) + batch_idx)
                self.tensorboard_writer.add_scalar('Accuracy/test', self.accuracy_average_test.get(), epoch * len(dataloader) + batch_idx)

            if with_logging:
                if batch_idx % self.log_interval == 0:
                    self.save(os.path.join(self.snapshot_folder_path, f"epoch_{epoch}_batch_{batch_idx}.pt"))
                    

        progress_bar.close()

        try:
            if is_train:
                if self.log_mlflow:
                    mlflow.log_metric("train_loss", self.loss_average_train.get(), step=epoch)
                    mlflow.log_metric("train_accuracy", self.accuracy_average_train.get(), step=epoch)
                    
            else:
                if self.log_mlflow:
                    mlflow.log_metric("test_loss", self.loss_average_test.get(), step=epoch)
                    mlflow.log_metric("test_accuracy", self.accuracy_average_test.get(), step=epoch)
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
        
    