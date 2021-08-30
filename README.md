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
Testing accuracy (\%) summarized below.
| Method | Actor |  Cornell | Texas | Wisconsin |
|:----|:---:|:---:|:---:|:---:|
| NEDA  | 38.17 |  87.78 | 87.50 | 89.40 |
| NEDA* | 38.01 |  87.22 | 88.06 | 89.80 |

## Usage
- To replicate the NEDA result on Actor, run the following script
```
python train_NEDA.py --no-cuda --dataset film --sample1 25 --sample2 10 --infected_number 25
```
- To replicate the NEDA* result on Actor, run the following script
```
python train_NEDA.py --no-cuda --is_copy --dataset film --sample1 25 --sample2 20 --infected_number 25
```
- To replicate the NEDA result on Cornell, run the following script
```
python train_NEDA.py --no-cuda --dataset cornell --sample1 24 --sample2 9 --infected_number 36
```
- To replicate the NEDA* result on Cornell, run the following script
```
python train_NEDA.py --no-cuda --is_copy --dataset cornell --sample1 18 --sample2 12 --infected_number 36
```
- To replicate the NEDA result on Texas, run the following script
```
python train_NEDA.py --no-cuda --dataset texas --sample1 24 --sample2 3 --infected_number 36
```
- To replicate the NEDA* result on Texas, run the following script
```
python train_NEDA.py --no-cuda --is_copy --dataset texas --sample1 18 --sample2 3 --infected_number 36
```
- To replicate the NEDA and NEDA* results on Wisconsin, run the following script
```
sh wisconsin.sh
```

## Acknowledgements
The original version of this code base was originally forked from https://github.com/williamleif/graphsage-simple/, and we owe many thanks to William L. Hamilton for making his code available.
