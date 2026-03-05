import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import copy
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

from builds.build_dataframes import BuildDataframes
from builds.build_datasets import BatteryDataset
from builds.standardization import Standardizer
from models import GRU, LSTM, MLP, CNN

class Exp:
    """
    This class encapsulates the entire experiment pipeline for training and 
    evaluating a surrogate model for battery SEI formation. It handles data 
    loading, preprocessing, model training, validation, testing, and results 
    saving. The class is designed to be flexible and configurable through 
    command-line arguments, allowing users to easily switch between different 
    models, datasets, and hyperparameters.
    """

    def __init__(self, configs, checkpoint_dir="checkpoints"):

        self.configs = configs
        self.device = torch.device(configs.device)

        if self.configs.mode != "test":

            # ----------------------------
            # Load raw data
            # ----------------------------
            train_dfs = BuildDataframes(configs.train_dir).get_dataframes()
            vali_dfs  = BuildDataframes(configs.vali_dir).get_dataframes()

            # ----------------------------
            # Standardize (fit ONLY on train)
            # ----------------------------
            self.scaler = Standardizer()
            self.scaler.fit(train_dfs)
            self.scaler.save(f"{self.configs.checkpoints_dir}/std_values.json")


            train_std = [self.scaler.transform(df) for df in train_dfs]
            vali_std  = [self.scaler.transform(df) for df in vali_dfs]
        
            # ----------------------------
            # Build train and vali datasets
            # ----------------------------
            train_dataset = BatteryDataset(train_std,
                                        window_size=configs.window_size,
                                        stride=configs.stride)
            vali_dataset  = BatteryDataset(vali_std,
                                        window_size=configs.window_size,
                                        stride=configs.stride)
            
            # ----------------------------
            # Build train and vali datasets
            # ----------------------------
            train_dataset = BatteryDataset(train_std,
                                        window_size=configs.window_size,
                                        stride=configs.stride)
            vali_dataset  = BatteryDataset(vali_std,
                                        window_size=configs.window_size,
                                        stride=configs.stride)
            
        
            # ----------------------------
            # DataLoaders
            # ----------------------------
            self.train_loader = DataLoader(train_dataset,
                                        batch_size=configs.batch_size,
                                        shuffle=True,
                                        drop_last=False)
            self.vali_loader = DataLoader(vali_dataset,
                                        batch_size=configs.batch_size,
                                        shuffle=False,
                                        drop_last=False)

        # ----------------------------
        # Build test Dataset and Dataloader
        # ----------------------------
    
        test_dfs  = BuildDataframes(configs.test_dir).get_dataframes()
        
        self.scaler = Standardizer()

        if self.configs.mode == "test":
            if self.configs.test_standardize_path is None:
                raise ValueError("Need to provide standardization json details.")
            else:
                self.scaler.load(self.configs.test_standardize_path)
        else:
            self.scaler.load(f"{self.configs.checkpoints_dir}/std_values.json")


        test_std  = [self.scaler.transform(df) for df in test_dfs]

       
        test_dataset  = BatteryDataset(test_std,
                                       window_size=configs.window_size,
                                       stride=1)

        
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=False)
        

        # ----------------------------
        # Model
        # ----------------------------
        self.model = self._load_model(configs.model).to(self.device)

        # ----------------------------
        # Optimizer
        # ----------------------------
        self.optimizer = self._select_optimizer()

        # ----------------------------
        # Loss
        # ----------------------------
        self.criterion = self._select_criterion()

        # Early stopping
        self.best_model = None
        self.best_val_loss = np.inf
        self.patience_counter = 0

    # =========================
    # Model Loader
    # =========================
    def _load_model(self, model_name):

        models = {
            "GRU": GRU,
            "LSTM": LSTM,
            "MLP": MLP,
            "CNN": CNN,
        }
        return models[model_name].Model(self.configs).float()

    # =========================
    # Optimizer
    # =========================
    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(),
                          lr=self.configs.learning_rate)

    # =========================
    # Loss
    # =========================
    def _select_criterion(self):

        if self.configs.loss == "mae":
            return nn.L1Loss()
        elif self.configs.loss == "mse":
            return nn.MSELoss()
        else:
            return nn.MSELoss()

    # =========================
    # Adaptive Learning Rate
    # =========================
    def _adjust_learning_rate(self, epoch):
        """
        lr_adjust = {epoch: max(lr_init if epoch <= 3 else lr_init * (0.5 ** (epoch - 3)), 1e-6)}
        """
        if epoch <= 3:
            lr = self.configs.learning_rate
        else:
            lr = self.configs.learning_rate * (0.5 ** (epoch - 3))
        lr = max(lr, 1e-6)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    # =========================
    # TRAIN LOOP
    # =========================
    def train(self):

        for epoch in range(0, self.configs.epochs):

            # adjust LR dynamically
            self._adjust_learning_rate(epoch)

            train_loss = self._train_one_epoch()
            val_loss   = self._validate()

            print(f"Epoch {epoch}/{self.configs.epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6e} |"
                  f"Patience Counter: {self.patience_counter}/{self.configs.patience}")

            # -------------------------
            # Early stopping
            # -------------------------
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0

                # save checkpoint
                checkpoint_path = os.path.join(self.configs.checkpoints_dir,
                                               f"best_model_{self.configs.model}.pt")
                torch.save(self.model.state_dict(), checkpoint_path)

            else:
                self.patience_counter += 1

            if self.patience_counter >= self.configs.patience:
                print("Early stopping triggered.")
                break

    # =========================
    # Single Train Epoch
    # =========================
    def _train_one_epoch(self):

        self.model.train()
        total_loss = 0

        for x, y in self.train_loader:

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    # =========================
    # Validation
    # =========================
    def _validate(self):

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in self.vali_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()

        return total_loss / len(self.vali_loader)

    # =========================
    # Test
    # =========================
    def test(self, checkpoint_path=None, inverse_transform=False):

        if checkpoint_path is not None:
           abc = torch.load(checkpoint_path)
           print(f"Checkpoint keys: {abc.keys()}")
           self.model.load_state_dict(torch.load(checkpoint_path))
           print()
        else:
            self.model.load_state_dict(torch.load(f"{self.configs.checkpoints_dir}/best_model_{self.configs.model}.pt"))
        print(f"Model Loaded.")

        self.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(self.device)
                pred = self.model(x)
                preds.append(pred.cpu().numpy())
                targets.append(y.numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        self.write_results(preds, targets)
        

        if inverse_transform:
            preds = self.scaler.inverse_transform_targets(preds.reshape(-1, preds.shape[-1]))
            targets = self.scaler.inverse_transform_targets(targets.reshape(-1, targets.shape[-1]))

        return preds, targets
    

    def write_results(self, preds, targets):

        # compute metrics per feature
        mse = mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        mse_sei = mean_squared_error(targets[:, 0], preds[:, 0])
        mse_temp = mean_squared_error(targets[:, 1], preds[:, 1])
        mae_sei = mean_absolute_error(targets[:, 0], preds[:, 0])
        mae_temp = mean_absolute_error(targets[:, 1], preds[:, 1])

        print(f"Test MSE: {mse:.6f}, MAE: {mae:.6f}, MSE_SEI: {mse_sei:.6f}, MAE_SEI: {mae_sei:.6f}, MSE_TEMP: {mse_temp:.6f}, MAE_TEMP: {mae_temp:.6f}")

        # -------------------------
        # Save to results.txt
        # -------------------------
        
        with open("results.txt", "a") as f:
            f.write("Battery Surrogate Model Test Metrics\n")
            f.write("-----------------------------------\n")
            f.write(f"MODEL:{self.configs.model}, SEED:{self.configs.seed}, MSE:{mse:.6f}, MAE:  {mae:.6f}, MSE_SEI: {mse_sei:.6f}, MAE_SEI: {mae_sei:.6f}, MSE_TEMP: {mse_temp:.6f}, MAE_TEMP: {mae_temp:.6f}")
            f.write(f"\n")
            f.write(f"\n")
            f.close()

    def save_results(self, preds, targets):
        np.save(f"{self.configs.results_dir}/preds.npy", preds)
        np.save(f"{self.configs.results_dir}/targets.npy", targets)