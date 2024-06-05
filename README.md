# Optimization and Distributed Training for Image Classification

This repository supports training ResNet18 on CIFAR-10 dataset along with 8 experimental evaluations of the training process. The experiments include:
- E1 **I/O Optimization**
- E2 **Profiling**
- E3 **Training on GPUs vs CPUs**
- E4: **Experimenting with Different Optimizers**
- E5: **Batch Size and Computational Efficiency**
- E6: **Multi-GPU Speedup Measurement**
- E7: **Computation vs Communication**
- E8: **Large Batch Training Accuracy**


## Set Up

To set up the experimental environment, run the following script:

```sh
sh scripts/setup.sh
```
## Running Experiments

For each of the above experiments, simply run the associated script. You can use the following script to run the first experiment - E1: 
```sh
sh scripts/E1.sh
```
You can modify the parameters and run experiments directly using `eval.puy`
```sh
python eval.py ----use_cuda=True --data_path='./data' --optimizer='sgd' --batch_size=64 --num_workers=2 --num_gpus=2 --get_bandwidth=True
```
