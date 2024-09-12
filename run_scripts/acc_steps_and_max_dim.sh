#!/usr/bin/env bash

python experiments/graph_alignment.py --dataset arenas --methods acc_ga_steps_and_dims --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5
python experiments/graph_alignment.py --dataset ppi_small --methods acc_ga_steps_and_dims --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5
python experiments/graph_alignment.py --dataset polblogs --methods acc_ga_steps_and_dims --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5
python experiments/graph_alignment.py --dataset enron_na --methods acc_ga_steps_and_dims --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5

python experiments/magna_graph_alignment.py --dataset magna --methods acc_ga_steps_and_dims --undirected 1 --seed 23423 --timeout 3600 --noise-level 15 --pp-mode whiten

python experiments/node_classification.py --dataset pyg_chameleon --methods acc_nc_steps_and_dims --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 3600 --pp-mode standardize --num-reps 1 --clf log_reg_sgd
python experiments/node_classification.py --dataset pyg_squirrel --methods acc_nc_steps_and_dims --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 3600 --pp-mode standardize --num-reps 1 --clf log_reg_sgd
python experiments/node_classification.py --dataset roman_empire --methods acc_nc_steps_and_dims --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 3600 --pp-mode standardize --num-reps 1 --clf log_reg_sgd
python experiments/node_classification.py --dataset ogb_arxiv_year --methods acc_nc_steps_and_dims --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 8000 --pp-mode standardize --num-reps 1 --clf log_reg_sgd


python experiments/graph_alignment.py --dataset arenas --methods sgcn_steps --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5
python experiments/graph_alignment.py --dataset ppi_small --methods sgcn_steps --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5
python experiments/graph_alignment.py --dataset polblogs --methods sgcn_steps --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5
python experiments/graph_alignment.py --dataset enron_na --methods sgcn_steps --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5

python experiments/magna_graph_alignment.py --dataset magna --methods sgcn_steps --undirected 1 --seed 23423 --timeout 3600 --noise-level 15 --pp-mode whiten

python experiments/node_classification.py --dataset pyg_chameleon --methods sgcn_steps --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 3600 --pp-mode standardize --num-reps 1 --clf log_reg_sgd
python experiments/node_classification.py --dataset pyg_squirrel --methods sgcn_steps --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 3600 --pp-mode standardize --num-reps 1 --clf log_reg_sgd
python experiments/node_classification.py --dataset roman_empire --methods sgcn_steps --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 3600 --pp-mode standardize --num-reps 1 --clf log_reg_sgd
python experiments/node_classification.py --dataset ogb_arxiv_year --methods sgcn_steps --undirected 0 --weighted 0 --node-attributed 1 --seed 89932 --timeout 8000 --pp-mode standardize --num-reps 1 --clf log_reg_sgd


