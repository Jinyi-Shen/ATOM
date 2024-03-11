## ATOM

This directory contains the code for ATOM, An Automatic Topology Synthesis Framework for Operational Amplifiers.

1. Install the circuit simulator hspice.

2. Install all code dependencies in the Python (recommended version 3.7) environment:

```pip install -r requirements.txt```

3. The code related to continuous topological representation learning is under the directory DVAE.
DVAE training:

```python train.py --model DVAE --batch_size 16 --n_topo 1600 --lr 1e-3 --trainset_size 1600 --epochs 500 --save-interval 50 --nz 10 --hs 500 --gpu 0 --save-appendix _1600```

The trained model is under DVAE/results.

DVAE testing:
```python train.py --only-test --model DVAE --n_topo 1600 --epochs 500 --load_model_name 500 --save-appendix _1600 --gpu 0```

4. The code related to freeze-thaw Bayesian optimization is under the directory FTBO.
Start the optimization for spec1 with
```./run.sh```

The topology optimization results for each run, including optimization time and the evaluation history of each visited topology, can be found under FTBO/pickle.

