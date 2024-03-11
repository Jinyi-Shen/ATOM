# ATOM: An Automatic Topology Synthesis Framework for Operational Amplifiers

This is the code repository for our proposed method ATOM that accompanys our paper ATOM: An Automatic Topology Synthesis Framework for Operational Amplifiers, which has been submitted to IEEE Transactions on Circuits and Systems II.

## Prerequisites and Dependencies

1. Install the circuit simulator hspice.

2. Install the prerequisite packages in ```./requirements.txt``` via ```pip``` or ```conda```. We used Anaconda Python 3.7 for our experiments.

## Running Experiments

1. The code related to continuous topological representation learning is under ```./DVAE/```.

DVAE training:

```python train.py --model DVAE --batch_size 16 --n_topo 1600 --lr 1e-3 --trainset_size 1600 --epochs 500 --save-interval 50 --nz 10 --hs 500 --gpu 0 --save-appendix _1600```

The trained model is under ```./DVAE/results/```.

DVAE testing:

```python train.py --only-test --model DVAE --n_topo 1600 --epochs 500 --load_model_name 500 --save-appendix _1600 --gpu 0```

2. The code related to freeze-thaw Bayesian optimization is under ```./FTBO/```.
Start the optimization for spec1 with

```./run.sh```

If optimization is required for spec2, please comment out the lines marked as "spec1" and uncomment the lines marked as "spec2" in ```./FTBO/TopoOpt_weibo.py``` and ```./FTBO/netlist_generator.py```

The topology optimization results for each run, including optimization time and the evaluation history of each visited topology, can be found under ```./FTBO/pickle/```.

