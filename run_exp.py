#!/usr/bin/env python
# run_exp.py

import argparse
import torch
from exp import Exp
import os
import random
import numpy as np
from utils.utilities import check_and_prepare_dirs, plot_predictions

parser = argparse.ArgumentParser(description="Battery Surrogate Experiment Runner")

# ---------------------------
# Mode
# ---------------------------
parser.add_argument("--mode", type=str, default="train",
                    choices=["train", "test"],
                    help="Run mode: train or test")
parser.add_argument("--test_model_folder", type=str, default="", help="If test mode, need to provide model.pt location.")


# ---------------------------
# Folder paths
# ---------------------------
parser.add_argument("--train_dir", type=str, default="Experiments/train", help="Path to training data")
parser.add_argument("--vali_dir", type=str, default="Experiments/vali", help="Path to validation data")
parser.add_argument("--test_dir", type=str, default="Experiments/test", help="Path to test data")
parser.add_argument("--plots_dir", type=str, default="plots", help="Directory to save plots")
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")


# ---------------------------
# Generic ML hyperparameters
# ---------------------------
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--num_workers", type=int, default=10, help="Number of workers for data loading")
parser.add_argument("--in_features", type=int, default=3, help="Number of input features")
parser.add_argument("--out_features", type=int, default=2, help="Number of output features")
parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--loss", type=str, default="mse", choices=["mse", "mae"], help="Loss function")
parser.add_argument("--model", type=str, default="MLP", help="Model type")
parser.add_argument("--window_size", type=int, default=30, help="lookback input window size")
parser.add_argument("--stride", type=int, default=1, help="Stride for sliding window")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device: cuda or cpu")

configs = parser.parse_args()

# Set random seeds for reproducibility
random.seed(configs.seed)
np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(configs.seed)

print("Running with configuration:")
print(configs)

settings = f"{configs.model}_{configs.seed}"
configs.checkpoints_dir = f"{configs.checkpoints_dir}/{settings}"
configs.results_dir = f"{configs.results_dir}/{settings}"
configs.plots_dir = f"{configs.plots_dir}/{settings}"

# Check and prepare directories
check_and_prepare_dirs(configs)

# ---------------------------
# Run training or testing
# ---------------------------
if configs.mode == "train":
    exp = Exp(configs)
    print("Starting training...")
    exp.train()

    # Evaluate on test set after training
    preds, targets = exp.test(inverse_transform=False)
    exp.save_results(preds, targets)
    plot_predictions(preds, targets, plots_dir=configs.plots_dir)

elif configs.mode == "test":

    print("Running test...")
    
    if configs.test_model_folder is None:
        raise ValueError("For testing, best model folder needs to be provided.")
    else:
        configs.test_standardize_path = os.path.join(configs.test_model_folder, "std_values.json")
        configs.test_model_path = os.path.join(configs.test_model_folder, f"best_model_{configs.model}.pt")

        exp = Exp(configs)
        preds, targets = exp.test(checkpoint_path=configs.test_model_path, inverse_transform=True)

    exp.save_results(preds, targets)
    plot_predictions(preds, targets, plots_dir=configs.plots_dir)

else:
    raise ValueError(f"Unknown mode: {configs.mode}")


