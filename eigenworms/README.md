# EigenWorms Experiments (Table 2)

This directory contains code we used for the EigenWorms experiments (Table 2).

This repository was originally forked from the official repository of the NRDE paper: [jambo6/neuralRDEs](https://github.com/jambo6/neuralRDEs).

## Requirements
The packages we used for our experiments can be found in the `requirements.txt` file.
See also the installation instructions of the original repo if needed: [jambo6/neuralRDEs](https://github.com/jambo6/neuralRDEs).

## Training and Evaluation
We used the following script to generate the numbers we report in Table 2
(the evaluation is automatically run at the end of training).

NB: When you run the script for the first time, the data will be automatically downloaded and processed. This may take some time.

The following script is expected to be called from `experiments` directory (`cd experiments`).

```
# seeds used for our results are: 1, 2, 3, 4, 5
SEED=1

# Results obtained for each seed are:
# 0.8974358974358975, 0.8717948717948718, 0.9743589743589743, 0.9230769230769231, 0.9230769230769231
# Mean (std): 91.8 (3.4)

# NB: the training size is 181 so the actual batch size is 181 even if we set it to 1024 below

python3 main_train.py \
  --seed ${SEED} \
  --depth 1 \
  --step 4 \
  --hidden_dim 128 \
  --num_heads 16 \
  --adjoint \
  --model_type logsig-ode-fwp \
  --report_every 1 \
  --learning_rule delta \
  --final_layer_scaling 10 \
  --batch_size 1024 \
  --trafo_ff_dim 64 \
  --lr 1e-4
```
