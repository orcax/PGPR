# Reinforcement Knowledge Graph Reasoning for Explainable Recommendation
This is code related to the SIGIR 2019 paper "Reinforcement Knowledge Graph Reasoning for Explainable Recommendation".


## Prepare Datasets
1. The datasets used in this paper can be download here: [https://www.dropbox.com/s/th672ttebwxhsfx/CIKM2017.zip?dl=0](https://www.dropbox.com/s/th672ttebwxhsfx/CIKM2017.zip?dl=0). The split is based on [1].
2. Assume the project directory is "/xxx/yyy/PGPR". Create a folder called "data" under the project directory (e.g. "/xxx/yyy/PGPR/data"). 
3. Place the downloaded dataset file (e.g. "CIKM2017.zip") under "data" folder (e.g. "/xxx/yyy/PGPR/data/CIKM2017.zip"). 
4. Unzip the zip file to extract a new dataset folder (e.g. "/xxx/yyy/PGPR/data/CIKM2017").


# Requirements
- Python >= 3.6
- PyTorch = 1.0


## How to run the code
1. Proprocess the data first:
```bash
python preprocess.py
```
2. Train knowledge graph embeddings (TransE in this case):
```bash
python train_transe_model.py --dataset [dataset_name]
```
""dataset_name"" should be one of "cd", "beauty", "clothing", "cell" (refers to utils.py)


## References
[1] Yongfeng Zhang, Qingyao Ai, Xu Chen, W. Bruce Croft. "Joint Representation Learning for Top-N Recommendation with Heterogeneous Information Sources". In Proceedings of CIKM. 2017.
