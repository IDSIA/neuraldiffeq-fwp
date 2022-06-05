# MuJoCo Experiments (Figure 1 in the appendix)

This directory contains code we used for the model-based reinforcement learning experiments in the appendix.

This repository was originally forked from the official repository of the baseline Latent ODE-RNN applied to model-based RL: [dtak/mbrl-smdp-ode](https://github.com/dtak/mbrl-smdp-ode)

## Requirements
The packages we used for our experiments can be found in the `requirements.txt` file.
See also the installation instructions of the original repo if needed: [dtak/mbrl-smdp-ode](https://github.com/dtak/mbrl-smdp-ode)
## Training and Evaluation
We used the following script to obtain the results we report in the paper. A log file will be created (under `logs`) for each run; search for `eval reward` in the file to check the evaluation performance.

```
# set MODEL to 'latent-ode' (baseline) or 'latent-ode-rfwp'
MODEL='latent-ode-rfwp'

# set MJC to 'hopper' or 'swimmer'
MJC='hopper'

# seeds used for our results are: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
SEED=1

python run.py \
  --mpc_ac \
  --adjoint \
  --obs_normal \
  --add_data_first \
  --model ${MODEL} \
  --env ${MJC} \
  --timer mlp \
  --epochs 200 \
  --gamma 0.99 \
  --lr 0.001 \
  --batch_size 128 \
  --eps_decay 0 \
  --latent_dim 128 \
  --max_steps 1000 \
  --mem_size 1000000 \
  --mb_epochs 80 \
  --mf_epochs 0 \
  --env_steps 5000 \
  --planning_horizon 10 \
  --ode_dim 128 \
  --ode_tol 1e-5 \
  --enc_hidden_to_latent_dim 20 \
  --num_restarts 0 \
  --seed ${SEED} \
  --log
```
