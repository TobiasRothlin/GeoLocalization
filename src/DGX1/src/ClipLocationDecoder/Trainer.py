import torch
import os
import mlflow   
import dotenv

from tqdm import tqdm

from MovingAverage import MovingAverage


        
        

class Trainer:

    def __init__(self,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss_function,
                 optimizer,
                 lr_scheduler,
                 gaussian_smoothing_scheduler,
                 gradient_accumulation_steps,
                 epochs,
                 device,
                 log_interval=10,
                 snapshot_path="./snapshots",
                 log_mlflow=False,
                 mlflow_experiment_name="GeoLocalization_Regression_Model"):
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.log_mlflow = log_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        self.gaussian_smoothing_scheduler = gaussian_smoothing_scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.snapshot_path = snapshot_path

        print(f"Using scheduler: {self.gaussian_smoothing_scheduler}")

        self.loss_average_train = MovingAverage(2*self.gradient_accumulation_steps)
        self.loss_average_test = MovingAverage(2*self.gradient_accumulation_steps)

        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)

        self.snapshot_folder_path = os.path.join(self.snapshot_path, "run")

        run_idx = 0
        while os.path.exists(self.snapshot_folder_path):
            self.snapshot_folder_path = os.path.join(self.snapshot_path, f"run_{run_idx}")
            run_idx += 1
        
        os.makedirs(self.snapshot_folder_path)

        self.training_log_file = os.path.join(self.snapshot_folder_path, "training.log")

        with open(self.training_log_file, "w") as f:
            f.write("")



    def __run_single_batch(self, batch,batch_idx,is_train):
        input_vec, targets = batch

        input_vec = input_vec.to(self.device)
        targets = targets.to(self.device)

        outputs = self.model(input_vec)

        loss = self.loss_function(outputs, targets)

        

        if is_train:
            self.loss_average_train.add(loss.item())

            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if batch_idx % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
        else:
            self.loss_average_test.add(loss.item())
    
    def __run_epoch(self, dataloader, epoch,with_logging=True,is_train=True):
        prefix = "Train" if is_train else "Test"
        progress_bar = tqdm(dataloader, desc=f"{prefix}Epoch {epoch}", postfix=f"Loss {self.loss_average_train.get():.5f}")
        for batch_idx, batch in enumerate(progress_bar):
            self.__run_single_batch(batch,batch_idx,is_train)

            if is_train:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_train.get():.5f}")
                self.log(f"Train,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_train.get():.5f}")
            else:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_test.get():.5f}")
                self.log(f"Test,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_test.get():.5f}")

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

            if self.gaussian_smoothing_scheduler is not None:
                self.gaussian_smoothing_scheduler.decay()

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


    def save(self, path):
        torch.save(self.model.state_dict(), path)


    def log(self, message):
        with open(self.training_log_file, "a") as f:
            f.write(message)
            f.write("\n")
        
    