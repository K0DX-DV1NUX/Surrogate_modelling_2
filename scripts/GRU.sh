#!/bin/sh

run_experiment() {
    mode=$1
    model=$2
    train_dir=$3
    vali_dir=$4
    test_dir=$5
    seed=$6

    python run_exp.py \
        --mode $mode \
        --model $model \
        --train_dir "$train_dir" \
        --vali_dir "$vali_dir" \
        --test_dir "$test_dir" \
        --plots_dir "plots" \
        --checkpoints_dir "checkpoints" \
        --results_dir "results" \
        --epochs 50 \
        --patience 10 \
        --batch_size 512 \
        --learning_rate 1e-3 \
        --loss mse \
        --num_workers 0 \
        --window_size 50 \
        --stride 5 \
        --seed $seed \
        --test_model_folder "checkpoints/${model}_${seed}"
}

# Example usage:
run_experiment train GRU Experiments/train Experiments/vali Experiments/test1 2021
run_experiment test GRU Experiments/train Experiments/vali Experiments/test2 2021
run_experiment test GRU Experiments/train Experiments/vali Experiments/test3 2021


run_experiment train GRU Experiments/train Experiments/vali Experiments/test1 2022
run_experiment test GRU Experiments/train Experiments/vali Experiments/test2 2022
run_experiment test GRU Experiments/train Experiments/vali Experiments/test3 2022

run_experiment train GRU Experiments/train Experiments/vali Experiments/test1 2023
run_experiment test GRU Experiments/train Experiments/vali Experiments/test2 2023
run_experiment test GRU Experiments/train Experiments/vali Experiments/test3 2023

run_experiment train GRU Experiments/train Experiments/vali Experiments/test1 2024
run_experiment test GRU Experiments/train Experiments/vali Experiments/test2 2024
run_experiment test GRU Experiments/train Experiments/vali Experiments/test3 2024

run_experiment train GRU Experiments/train Experiments/vali Experiments/test1 2025
run_experiment test GRU Experiments/train Experiments/vali Experiments/test2 2025
run_experiment test GRU Experiments/train Experiments/vali Experiments/test3 2025
