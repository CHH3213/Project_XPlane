# ddpg-aigym

## Deep Deterministic Policy Gradient
Implementation of Deep Deterministic Policy Gradiet Algorithm (Lillicrap et al.[arXiv:1509.02971](http://arxiv.org/abs/1509.02971).) in Tensorflow

## How to use
```
cd ddpg-aigym
python main.py
```
## Features
- Batch Normalization (improvement in learning speed)
- Grad-inverter (given in arXiv: [arXiv:1511.04143](http://arxiv.org/abs/1511.04143))

To use batch normalization
```
is_batch_norm = True #batch normalization switch
```

dependicies:
 tensorflow =2.x, openaiGym=0.10.5, python=3.8





