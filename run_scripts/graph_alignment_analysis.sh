#!/usr/bin/env bash

python experiments/graph_alignment_per_node_analysis.py --dataset arenas --undirected 1 --methods detailed_ga_analysis --noise-p 0.15 --pp-mode none::whiten --dims 512
python experiments/graph_alignment_per_node_analysis.py --dataset ppi_small --undirected 1 --methods detailed_ga_analysis --noise-p 0.15 --pp-mode none::whiten --dims 512
python experiments/graph_alignment_per_node_analysis.py --dataset magna --undirected 1 --methods detailed_ga_analysis --noise-p 0.15 --pp-mode none::whiten --dims 512
python experiments/graph_alignment_per_node_analysis.py --dataset enron_na --undirected 0 --methods detailed_ga_analysis --noise-p 0.15 --pp-mode none::whiten --dims 512
python experiments/graph_alignment_per_node_analysis.py --dataset polblogs --undirected 0 --methods detailed_ga_analysis --noise-p 0.15 --pp-mode none::whiten --dims 512
python experiments/graph_alignment_per_node_analysis.py --dataset polblogs --undirected 0 --methods detailed_ga_analysis12 --noise-p 0.15 --pp-mode none::whiten --dims 512

