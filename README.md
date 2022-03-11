## NEDA: A Novel Message Passing Neural Network for Disassortative Network Representation Learning
#### Author: [Yanfeng Xue] (xueyanfeng0819@qq.com)

## Dependencies
- python 3.8.8
- pytorch 1.8.0
- networkx 2.5.1
- scikit-learn 0.24.2

## Datasets
The `data` folder contains four benchmark datasets (Actor, i.e., film, Cornell, Texas and Wisconsin) from [Geom-GCN](https://github.com/graphdml-uiuc-jlu/geom-gcn).

## Results
1.   Node classification accuracy(\%) in the 60%, 20% and 20% splits for training, validation, and test splits. 
| Method | Actor |  Cornell | Texas | Wisconsin |
|:----|:---:|:---:|:---:|:---:|
| NEDA  | 38.17 |  87.78 | 87.50 | 89.40 |
| NEDA* | 38.01 |  87.22 | 88.06 | 89.80 |

2 Node classification accuracy(\%) in the 48%, 32% and 20% splits for training, validation, and test splits. 
| Method | Actor |  Cornell | Texas | Wisconsin |
|:----|:---:|:---:|:---:|:---:|
| NEDA  | 37.57 |  86.39 | 86.11 | 88.40 |
| NEDA* | 37.64 |  86.39 | 85.83 | 88.40 |

## Usage
- To replicate the NEDA and NEDA* results on Actor, run the following script
```
sh film.sh
```
- To replicate the NEDA and NEDA* results on Cornell, run the following script
```
sh cornell.sh
```
- To replicate the NEDA and NEDA* results on Texas, run the following script
```
sh texas.sh
```
- To replicate the NEDA and NEDA* results on Wisconsin, run the following script
```
sh wisconsin.sh
```

## Acknowledgements
The original version of this code base was originally forked from https://github.com/williamleif/graphsage-simple/, and we owe many thanks to William L. Hamilton for making his code available.
