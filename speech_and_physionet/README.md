
# Speech Commands & PhysioNet Sepsis Experiments (Table 1)

This directory contains code we used for the Speech Commands and PhysioNet Sepsis datasets (Table 1).

This repository was originally forked from the official repository of the NCDE paper: [patrick-kidger/NeuralCDE](https://github.com/patrick-kidger/NeuralCDE).

## Requirements
The requirements are the same as those of the original NCDE paper, except that our code also supports the latest versions of torchaudio.
The packages/versions we used for our experiments can be found in the `requirements.txt` file.

## Training and Evaluation
We used the following scripts to generate the numbers we report in Table 1
(the evaluation is automatically run at the end of training).

NB: When you run the script for the first time, the data will be automatically downloaded and processed.
This may take some time.

### Speech Commands
```
# NB: `hidden_hidden_channels` and `num_hidden_layers` are set to dummy numbers as they are irrelevant for delta_cde

# seeds used for our results are: 1, 2, 3, 4, 5
SEED=1

python3 experiments/main.py \
  --name_task speech_commands \
  --max_epochs 500 \
  --seed ${SEED} \
  --model_name delta_cde \
  --hidden_channels 80 \
  --hidden_hidden_channels 1 \
  --num_hidden_layers 3 \
  --num_heads 4 \
  --batch_size 1024 \
  --base_lr 5e-05 \
  --grad_scale 100 \
  --trafo_ff_dim 320
```

### PhysioNet Sepsis, "no-OI"

```
# NB: `hidden_hidden_channels` and `num_hidden_layers` are set to dummy numbers as they are irrelevant for delta_cde

# seeds used for our results are: 1, 2, 3, 4, 5
SEED=1

python3 experiments/main.py \
  --name_task sepsis \
  --seed ${SEED} \
  --max_epochs 200 \
  --model_name delta_cde \
  --hidden_channels 160 \
  --hidden_hidden_channels 1 \
  --num_hidden_layers 3 \
  --num_heads 32 \
  --batch_size 1024 \
  --base_lr 4e-05 \
  --grad_scale 100 \
  --cde_use_v_laynorm
```

### PhysioNet Sepsis, "OI"

The flag `--use_sepsis_intensity` should be added for the 'OI' version of the task.

```
# NB: `hidden_hidden_channels` and `num_hidden_layers` are set to dummy numbers as they are irrelevant for delta_cde

# seeds used for our results are: 1, 2, 3, 4, 5
SEED=1

python3 experiments/main.py \
  --name_task sepsis \
  --use_sepsis_intensity \
  --seed ${SEED} \
  --max_epochs 200 \
  --model_name delta_cde \
  --hidden_channels 80 \
  --hidden_hidden_channels 1 \
  --num_hidden_layers 3 \
  --num_heads 16 \
  --trafo_ff_dim 64 \
  --batch_size 1024 \
  --base_lr 3e-05 \
  --delta_ode_post_tahn \
  --grad_scale 100
```
