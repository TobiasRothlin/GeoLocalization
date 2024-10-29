import torch
from tqdm import tqdm

import mlflow   

import dotenv

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8890"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer:

    def __init__(self, model,
                    train_loader,
                    test_loader, 
                    save_every, 
                    snapshot_path,
                    loss_function, 
                    gpu_id,
                    gradient_accumulation_steps=1,
                    lr=1e-6,
                    amsgrad=True,
                    weight_decay=1e-4,
                    betas=(0.9,0.98),
                    gamma=0.9):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_every = save_every
        self.snapshot_path = snapshot_path

        self.current_loss_history = []
        self.max_loss_history = 2*gradient_accumulation_steps

        self.use_mlflow = True

        if gpu_id == 0:
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
        
        self.loss_function = loss_function
        self.gpu_id = gpu_id
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = self.model.to(self.gpu_id)
        self.model = DDP(model, device_ids=[gpu_id],find_unused_parameters=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=amsgrad, weight_decay=weight_decay, betas=betas)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        print(f"Trainer initialized on GPU {self.gpu_id}")


    def _run_batch(self, source, targets, batch_number):
        outputs = self.model(source)
        loss = self.loss_function(outputs, targets)

        self.current_loss_history.append(loss.item())

        if len(self.current_loss_history) > self.max_loss_history:
            self.current_loss_history.pop(0)
        
        # Scale the loss by accumulation steps
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        # Perform optimizer step and zero gradients after accumulation steps
        if (batch_number + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        
        
        return loss.item()


    def _run_epoch(self, epoch):
        self.train_loader.sampler.set_epoch(epoch)
        batch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"  Epoch {epoch} on [{self.gpu_id}]", postfix=f"Loss: {batch_loss:.5f}, Average Loss: 0",position=self.gpu_id*(epoch+1) )
        
        batch_number = 0
        self.optimizer.zero_grad()
        for images, labels in progress_bar:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                batch_loss = self._run_batch(images, labels,batch_number)

                if self.gpu_id == 0:
                    if self.use_mlflow:
                        try:
                            mlflow.log_metric("Loss", batch_loss)
                            mlflow.log_metric("Average Loss", sum(self.current_loss_history)/len(self.current_loss_history))
                        except Exception as e:
                            print("Could not connect to MLFlow")
                            print(e)
                            self.use_mlflow = False
                            print("MLFlow disabled")

                    if batch_number % self.save_every == 0:
                        self.save(batch_number,epoch)

                progress_bar.set_postfix_str(f"Loss: {batch_loss:.5f} , Average Loss: {sum(self.current_loss_history)/len(self.current_loss_history):.5f}")
                batch_number += 1

        progress_bar.close()

    def run_test(self):
        self.model.eval()
        epoch_loss = 0
        progress_bar = tqdm(self.test_loader, desc=f"  Testing on [{self.gpu_id}]", postfix=f"Loss: {epoch_loss:.5f}",position=self.gpu_id)
        
        batch_number = 0
        for images, labels in progress_bar:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                with torch.no_grad():
                    outputs = self.model(images)
                    loss = self.loss_function(outputs, labels)
                    epoch_loss = loss.item()
            
                progress_bar.set_postfix_str(f"Loss: {epoch_loss:.5f}")
                batch_number += 1

        progress_bar.close()
        self.model.train()

        return epoch_loss

    def train(self, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            self._run_epoch(epoch)
            self.scheduler.step()

        self.save(name=self.run_path+"/model_final.pt")

    def save(self,idx=0,epoch=0,name=None):
        if name:
            print(f"Saving model checkpoint to {name}")
            torch.save(self.model.module.state_dict(), name) 
            return name
        else:
            print(f"Saving model checkpoint to {self.run_path}")
            torch.save(self.model.module.state_dict(), self.run_path+"/"+f"model_{epoch}_{idx}.pt") 
            return self.run_path+"/"+f"model_{epoch}_{idx}.pt"  

        
        


    