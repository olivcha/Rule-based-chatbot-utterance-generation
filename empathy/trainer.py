# import the necessary packages and modules
from tqdm import tqdm # for visualising progress bar of training process
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import datetime
import torch
import logging
from functools import lru_cache
import gc
from typing import List
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os


# import from the other files
from model import EmpathyClassificationModel
from utils import labels, TokenizersCollateFn, EarlyStopper, EmpathyDataset, Mish


class Trainer():
    ''' Class trainer implemented with steps for model training/finetuning task.
      The new model uses a roberta base model and uses the empathy classifier
      finetuning steps. Cross entropy loss is used for the multi-class classification
      task. Early stopping is implemented.
    '''

    def __init__(self, train_path, val_path, test_path, location, batch_size=20, epochs=10, lr=2E-06, second_ft=False):
        '''
        Constructor for the Trainer class. 
            Args:
                batch_size - batch size for training
                epochs - number of epochs for training
                lr - learning rate for training
        '''
        super().__init__()
        
        # load the model - roberta base model with empathy classifier
        self.model = EmpathyClassificationModel(AutoModelWithLMHead.from_pretrained("roberta-base").base_model, len(labels))
        self.loss = nn.CrossEntropyLoss() #cross entropy loss since this is multi-class classification
        self.loss.require_grad = True
        self.lr = lr
        self.batch_size = batch_size
        self.warmup_steps = 100
        self.epochs = epochs
        self.accumulate_grad_batches = 1
        
        # paths for the data
        self.train_path=train_path
        self.val_path=val_path
        self.test_path=test_path
        self.location = location
        
        # initialize the early stopper for the model
        self.early_stopper = EarlyStopper(patience=20, verbose=True, min_delta=0.001)
        self.optimizer = None
        self.scheduler = None
        self.second_ft = second_ft
        
        # initialise the loss lists and validation metric
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_metric = torchmetrics.Accuracy(num_classes=len(labels), task='multiclass')
        
        # move variables to device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)
            self.val_metric = self.val_metric.to(self.device)
        else:
            self.device = torch.device('cpu')

    def fit(self):
        ''' 
        Training function for the model. 
            Args:
                self
            Returns:
                None
        '''
        
        # load the data
        train_dataloader = self.train_dataloader()
        val_dataloader = self.val_dataloader()

        # configure optimizers
        optimizers = self.configure_optimizers()
        self.optimizer = optimizers['optimizer']
        self.scheduler = optimizers['lr_scheduler']
        
        # keeping track of the best model
        best_val_loss = float('inf') # set to infinity
        if self.second_ft:
            best_model_path = os.path.join("saved_models", "best_model", "best_model_second_ft.pt")
        else:
            best_model_path = os.path.join("saved_models", "best_model", "best_model_first_ft.pt")

        # log each epoch
        logging.info(f"Training of empathy classifier started at {datetime.datetime.now()}, optimizer: {self.optimizer}, scheduler: {self.scheduler}, learning rate: {self.lr}, loss function: {self.loss}")
        
        # training - iterate through epochs with progress bar
        for epoch in tqdm(range(self.epochs)):  # loop over the dataset multiple times 
            self.model.train()
            train_loss = 0.0 
             
            # iterate through batches using dataloader
            for batch_idx, batch in enumerate(train_dataloader):
                batch = self.to_device(batch)  # move batch tensors to device
                loss = self.training_step(batch, batch_idx=batch_idx)
                train_loss += loss
            
            # calculate average loss 
            avg_train_loss = train_loss / len(train_dataloader)
            self.train_loss_history.append(avg_train_loss)
            
            # validation 
            self.model.eval()
            val_loss = 0.0
            val_metric = 0.0
            
            # iterate through batches using dataloader for validation
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    batch = self.to_device(batch)  # move batch tensors to device
                    loss = self.validation_step(batch, batch_idx=batch_idx)
                    val_loss += loss
            
            # calculate average loss and metric (accuracy)
            avg_val_loss = val_loss / len(val_dataloader)
            self.val_loss_history.append(avg_val_loss)
            val_accuracy = self.val_metric.compute()
            
            self.early_stopper(val_loss, self.model)
            
            logging.info(f"Epoch {epoch + 1}/{self.epochs} - "
                     f"Train Loss: {avg_train_loss:.4f} - "
                     f"Val Loss: {avg_val_loss:.4f} - "
                     f"Val Accuracy: {val_accuracy:.4f}")
            
            print(f"Epoch {epoch + 1}/{self.epochs} - "
                     f"Train Loss: {avg_train_loss:.4f} - "
                     f"Val Loss: {avg_val_loss:.4f} - "
                     f"Val Accuracy: {val_accuracy:.4f}")
            
            if avg_val_loss < best_val_loss:
                # update best validation loss
                best_val_loss = avg_val_loss
                # save the best model
                self.save_model(best_model_path)
            
            # early stopping if validation loss does not improve 
            if self.early_stopper.early_stop:
                logging.info(f"Early stopping at epoch {epoch + 1} with validation loss: {avg_val_loss:.4f}")
                break
   
            self.save_model(os.path.join("saved_models", str(epoch) + ".pt"))                    

    def to_device(self, batch):
        '''
        Move batch to device (CPU/GPU).
            Args:
                batch - batch of data
            Returns:
                batch - batch of data on device                
        '''
        if isinstance(batch, tuple): # if batch is tuple 
            return tuple(self.to_device(b) for b in batch)
        elif isinstance(batch, list):  # if batch is list
            return [self.to_device(b) for b in batch]
        elif isinstance(batch, dict): # if batch is dictionary
            return {key: self.to_device(value) for key, value in batch.items()}
        else: # if batch is tensor
            return batch.to(self.device)

    def evaluate(self):
        '''
        Evaluation function for the model.
            Args:
                self
            Returns:
                test_loss - test loss
        '''
        test_dataloader = self.test_dataloader()
        print("Running test set evaluation...")
        logging.info("Running test set evaluation...")
        
        self.model.eval().to(self.device)
        true_y, pred_y = [], []
        total_loss = 0.0
        total_samples = 0
        
        # iterate through batches using dataloader
        with torch.no_grad(): # no gradient calculation
            for i, batch_ in enumerate(test_dataloader):
                (X, attn), y = batch_ 
                
                batch = (X.cuda(), attn.cuda())
                logging.info(f"Testing Progress: {i + 1}/{len(test_dataloader)}")
                y_pred = torch.argmax(self.model(batch), dim=1)
                                
                true_y.extend(y.cpu()) 
                pred_y.extend(y_pred.cpu())

                total_samples += y.size(0)
                
        # Print classification report and confusion matrix
        print("\n" + "_" * 80)
        print(total_samples)
        # print classification report and confusion matrix
        print(classification_report(true_y, pred_y, target_names=labels, digits=4))
        logging.info(classification_report(true_y, pred_y, target_names=labels, digits=4))

        # Plot confusion matrix
        cm = confusion_matrix(true_y, pred_y, labels=range(len(labels)))
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)

        # plot confusion matrix
        plt.rcParams.update({'font.size': 12})
        plt.figure(figsize=(10, 8))
        sn.heatmap(df_cm, annot=True, cmap='Greens', fmt='g')
        plt.show()
        plt.savefig("confusion_matrix.png")
                
    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx):
        ''' 
        Training step for the model.
            Args:
                self
                batch - batch of data
                batch_idx - batch index
            Returns:
                loss - loss value
        '''
        inputs, labels = batch
        inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
        labels = labels.to(self.device)
        
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
        # return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        ''' 
        Validation step for the model.
            Args:
                self
                batch - batch of data   
                batch_idx - batch index
            Returns:
                loss - loss value
        '''
        inputs, labels = batch
        inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
        labels = labels.to(self.device)
        
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        self.val_metric(outputs, labels)
        
        return loss.item()
        # return self.step(batch, "val")

    def validation_end(self, outputs: List[dict]):
        '''
        Validation end function for the model.
            Args:
                self
                outputs - list of outputs
            Returns:
                val_loss - validation loss 
        '''
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        ''' 
        Test step for the model.
            Args:
                self
                batch - batch of data
                batch_idx - batch index
            Returns:
                loss - loss value
        '''
        inputs, labels = batch
        inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
        labels = labels.to(self.device)
        
        outputs = self.forward(inputs)
        loss = self.loss(outputs, labels)
        print(loss)
        logging.info("Test loss: " + str(loss))
        self.test_metric(outputs, labels)  # Update the test metric
        
        return loss.item()
        # return self.step(batch, "test")

    def train_dataloader(self):
        ''' 
        Train dataloader for the model.
            Args:
                self
            Returns:
                dataloader - dataloader for the model with train data
        '''
        return self.create_data_loader(self.train_path, shuffle=True)

    def val_dataloader(self):
        ''' 
        Validation dataloader for the model.
            Args:
                self
            Returns:
                dataloader - dataloader for the model with validation data
        '''
        return self.create_data_loader(self.val_path)

    def test_dataloader(self):
        ''' 
        Test dataloader for the model.
            Args:
                self
            Returns:
                dataloader - dataloader for the model with test data
        '''
        return self.create_data_loader(self.test_path)

    def create_data_loader(self, ds_path: str, shuffle=False):
        ''' 
        Create dataloader for the model.
            Args:
                self
                ds_path - path to dataset
                shuffle - shuffle data
            Returns:
                dataloader - dataloader for the model                 
        ''' 
        return DataLoader(
                    EmpathyDataset(ds_path),
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    collate_fn=TokenizersCollateFn()
        )

    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.accumulate_grad_batches * self.epochs

    def configure_optimizers(self):
        ''' 
        Optimizer configuration for the model.
            Args:
                self
            Returns:
                optimizer - optimizer
                lr_scheduler - learning rate scheduler
        ''' 
        optimizer = AdamW(self.model.parameters(), eps=1e-08, lr=self.lr) #we use AdamW as this usually performs well
        
        # the learning rate scheduler we use is the one with linear schedule with warmup
        lr_scheduler = get_linear_schedule_with_warmup( 
                    optimizer,
                    num_warmup_steps=self.warmup_steps,
                    num_training_steps=self.total_steps(),
        )
        self.optimizer = optimizer
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        # return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def save_model(self, path):
        ''' 
        Saving model.
            Args:
                self
                path - path to save model
        '''
        torch.save(self.model.state_dict(), path)