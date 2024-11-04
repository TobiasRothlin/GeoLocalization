import os
import torch
import json
import mlflow
from tqdm import tqdm
import dotenv

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from MovingAverage import MovingAverage

def __setup_ddp(rank,world_size):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "8890"
        torch.cuda.set_device(rank)
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

def temp(rank,world_size):
    print(f"--> Starting Training on GPU {rank}, World Size: {world_size}")

def run(rank,world_size,
        config,
        train_dataset,
        test_dataset,
        model,
        loss_function,
        optimizer,
        lr_scheduler):
    
    print(f"--> Starting Training on GPU {rank}, World Size: {world_size}")
    __setup_ddp(rank,world_size)

    train_dataloader = DataLoader(train_dataset,sampler=DistributedSampler(train_dataset), **config["DataLoaderConfig"]["Train"])
    test_dataloader = DataLoader(test_dataset, **config["DataLoaderConfig"]["Test"])

    trainer = MultiGPUTrainer(train_dataloader=train_dataloader,test_dataloader=test_dataloader,model=model,loss_function=loss_function,optimizer=optimizer,lr_scheduler=lr_scheduler,epochs=config["TrainingConfig"]["Epochs"],device=rank,log_interval=config["TrainingConfig"]["SaveEvery"],snapshot_path=config["TrainingConfig"]["SnapshotPath"],log_mlflow=config["TrainingConfig"]["LogMLFlow"],mlflow_experiment_name=config["TrainingConfig"]["MLFlowExperimentName"],full_run_config=config)
    trainer.train()

    destroy_process_group()

def MultiGPUTraining(config,
                        train_dataset,
                        test_dataset,
                        model,
                        loss_function,
                        optimizer,
                        lr_scheduler):
     
    world_size = torch.cuda.device_count()
    print(f"String Training on {world_size} GPUs")
    mp.spawn(run,
                args=(world_size,
                        config,
                        train_dataset,
                        test_dataset,
                        model,
                        loss_function,
                        optimizer,
                        lr_scheduler),
                nprocs=world_size)
    
    
     

    




class MultiGPUTrainer():

    def __init__(self,
                train_dataloader,
                test_dataloader,
                model,
                loss_function,
                optimizer,
                lr_scheduler,
                epochs,
                device,
                log_interval,
                snapshot_path,
                log_mlflow,
                mlflow_experiment_name,
                full_run_config):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.device = device
        self.log_interval = log_interval
        self.snapshot_path = snapshot_path
        self.log_mlflow = log_mlflow
        self.mlflow_experiment_name = mlflow_experiment_name
        self.full_run_config = full_run_config
        

        self.model = self.model.to(self.device)
        self.model = DDP(model, device_ids=[device],find_unused_parameters=True)

        if self.device == 0:
            self.__run_setup()

        print(f"Trainer initialized on GPU {self.gpu_id}")


    def __run_setup(self):
        if not os.path.exists(self.snapshot_path):
            os.makedirs(self.snapshot_path)


        self.run_path = os.path.join(self.snapshot_path, "run")
        new_path = self.run_path
        idx = 0
        while os.path.exists(new_path):
            new_path = self.run_path + "_" + str(idx)
            idx += 1
        
        self.run_path = new_path
        os.makedirs(self.run_path)

        if self.full_run_config is not None:
            with open(os.path.join(self.snapshot_folder_path, "run_config.json"), "w") as f:
                json.dump(self.full_run_config, f, indent=4)

        self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(self.snapshot_folder_path, "tensorboard"))
        self.training_log_file = os.path.join(self.snapshot_folder_path, "training.log")

        model_input,model_output =self.test_dataloader.dataset[0]

        model_input = model_input.unsqueeze(0).to(self.device)
        model_output = model_output.unsqueeze(0).to(self.device)

        self.tensorboard_writer.add_graph(self.model, model_input)

        print(f"Snapshot folder: {self.snapshot_folder_path}")

        with open(self.training_log_file, "w") as f:
            f.write("")

        self.loss_average_train = MovingAverage(1)
        self.loss_average_test = MovingAverage(1)


    def __run_single_batch(self, batch,is_train):
        if is_train:
            self.optimizer.zero_grad()

        input_vec, targets = batch
        input_vec = input_vec.to(self.device)
        targets = targets.to(self.device)

        embedding = self.model.module.get_embedding(input_vec)

        outputs = self.model(embedding)

        loss = self.loss_function(outputs, targets)

        if is_train:
            self.loss_average_train.add(loss.item())
            loss.backward()
            self.optimizer.step()
        else:
            self.loss_average_test.add(loss.item())


    def __run_epoch(self, dataloader, epoch,with_logging=True,is_train=True):
        prefix = "Train" if is_train else "Test"
        progress_bar = tqdm(dataloader, desc=f"{prefix}Epoch {epoch}", postfix=f"Loss {self.loss_average_train.get():.5f}")
        for batch_idx, batch in enumerate(progress_bar):
            self.__run_single_batch(batch,batch_idx,is_train)

            if is_train:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_train.get():.5f}")
                if self.device == 0:
                    self.log(f"Train,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_train.get():.5f}")
                    self.tensorboard_writer.add_scalar('Loss/train', self.loss_average_train.get(), epoch * len(dataloader) + batch_idx)
            else:
                progress_bar.set_postfix_str(f"Loss {self.loss_average_test.get():.5f}")
                if self.device == 0:
                    self.log(f"Test,Epoch:{epoch},Batch:{batch_idx},Loss:{self.loss_average_test.get():.5f}")
                    self.tensorboard_writer.add_scalar('Loss/test', self.loss_average_test.get(), epoch * len(dataloader) + batch_idx)
            
            if with_logging and self.device == 0:
                if batch_idx % self.log_interval == 0:
                    self.save(os.path.join(self.snapshot_folder_path, f"epoch_{epoch}_batch_{batch_idx}.pt"))
                    

        progress_bar.close()
        if self.device == 0:
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
        if self.device == 0:
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
            if self.device == 0:
                self.save(os.path.join(self.snapshot_folder_path, f"model_end_of_epoch_{epoch}.pt"))

        if self.device == 0:
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
        state_dict = self.model.module.state_dict()
        torch.save(state_dict, path)

    def log(self, message):
        with open(self.training_log_file, "a") as f:
            f.write(message)
            f.write("\n")
            

        

    

        