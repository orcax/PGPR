# Reinforcement Knowledge Graph Reasoning for Explainable Recommendation
This is code related to the SIGIR 2019 paper "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation".


## Datasets


## Requirements
- Python >= 3.6
- PyTorch = 1.0


## How to run the code
1. Proprocess the data first:
```bash
python preprocess.py --dataset <dataset_name>
```
2. Train knowledge graph embeddings (TransE in this case):
```bash
python train_transe_model.py --dataset <dataset_name>
```
"<dataset_name>" should be one of "cd", "beauty", "clothing", "cell" (refer to utils.py).


## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft. "Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources". In Proceedings of CIKM. 2017.
