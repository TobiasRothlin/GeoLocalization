import torch
from tqdm import tqdm

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

    def __init__(self, model, train_loader, test_loader, optimizer, save_every, snapshot_path,loss_function, gpu_id,gradient_accumulation_steps=1):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.save_every = save_every
        self.snapshot_path = snapshot_path
        self.loss_function = loss_function
        self.gpu_id = gpu_id
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.model = self.model.to(self.gpu_id)
        self.model = DDP(model, device_ids=[gpu_id],find_unused_parameters=True)

        if self.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-6,amsgrad=True,weight_decay=1e-4,betas=(0.9,0.98))
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)

        print(f"Trainer initialized on GPU {self.gpu_id}")


    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        outputs = self.model(source)
        loss = self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()


    def _run_epoch(self, epoch):
        self.train_loader.sampler.set_epoch(epoch)
        epoch_loss = 0
        progress_bar = tqdm(self.train_loader, desc=f"  Epoch {epoch} on [{self.gpu_id}]", postfix=f"Loss: {epoch_loss:.5f}",position=self.gpu_id )
        
        batch_number = 0
        for images, labels in progress_bar:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                epoch_loss = self._run_batch(images, labels)

                if self.gpu_id == 0 and batch_number % self.save_every == 0:
                    self.save(batch_number)

                progress_bar.set_postfix_str(f"Loss: {epoch_loss:.5f}")
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
                    epoch_loss += loss.item() / len(self.test_loader)
            
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


    def save(self,idx=0):
        print(f"Saving model checkpoint to {self.snapshot_path}")
        torch.save(self.model.state_dict(), self.snapshot_path+"/"+f"model_{idx}.pt")   

        
        


    