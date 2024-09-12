#!/usr/bin/env bash

python experiments/graph_alignment.py --dataset arenas --methods acc_vs_pcapass_k4_k10 --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode none::whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset ppi_small --methods acc_vs_pcapass_k4_k10 --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode none::whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset polblogs --methods acc_vs_pcapass_k4_k10 --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode none::whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset enron_na --methods acc_vs_pcapass_k4_k10 --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode none::whiten --num-reps 5 --dims 512

python experiments/magna_graph_alignment.py --dataset magna --methods acc_vs_pcapass_k4_k10 --undirected 1 --seed 23423 --timeout 3600 --noise-level 15 --pp-mode none::whiten --dims 512

python experiments/graph_alignment.py --dataset arenas --methods acc_rtol_sweep pcapass_rtol_sweep --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset ppi_small --methods acc_rtol_sweep pcapass_rtol_sweep --undirected 1 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset polblogs --methods acc_rtol_sweep4 pcapass_rtol_sweep4 --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
python experiments/graph_alignment.py --dataset enron_na --methods acc_rtol_sweep4 pcapass_rtol_sweep4 --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512

python experiments/magna_graph_alignment.py --dataset magna --methods acc_rtol_sweep pcapass_rtol_sweep --undirected 1 --seed 23423 --timeout 3600 --noise-level 15 --pp-mode whiten --dims 512

python experiments/graph_alignment.py --dataset polblogs --methods acc_rtol_sweep12 --undirected 0 --seed 23423 --timeout 3600 --noise-p 0.15 --pp-mode whiten --num-reps 5 --dims 512
