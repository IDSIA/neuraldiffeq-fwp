import os
import sys
import json
import argparse
import random
import time
from datetime import datetime
import numpy as np
import torch
from mbrl import MBRL
import utils

from envs.windygrid_simulator import WindyGridSimulator
from envs.hiv_simulator import HIVSimulator
from envs.acrobot_simulator import AcrobotSimulator
try:
    from envs.half_cheetah_simulator import HalfCheetahSimulator
    from envs.swimmer_simulator import SwimmerSimulator
    from envs.hopper_simulator import HopperSimulator
except:
    print("Couldn't import Mujoco.")


parser = argparse.ArgumentParser('Running model-based RL')
parser.add_argument('--train_env_model', action='store_true',
                    help='train environment model')
parser.add_argument('--world_model', action='store_true',
                    help='learn the world model')
parser.add_argument('--latent_policy', action='store_true',
                    help='whether make decision based on latent variables')  # Do not use. Not well tested.
parser.add_argument('--num_restarts', type=int, default=0,
                    help='the number of restarts')
parser.add_argument('--restart_from', type=str, default='',
                    help='path of checkpoints to load from')  # Do not use. Not well tested.
parser.add_argument('--save_all', action='store_true',
                    help='save things for possible restarting')
parser.add_argument('--model', type=str, default='free',
                    help='the environment model, or load the training model')
parser.add_argument('--trained_model_path', type=str, default='',
                    help='the pre-trained environment model path')
parser.add_argument('--env', type=str, default='acrobot',
                    help='the environment')
parser.add_argument('--timer', type=str, default='fool',
                    help='the type of timer')
parser.add_argument('--seed', type=int, default=0,
                    help='the random seed')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='the discount factor')
parser.add_argument('--obs_normal', action='store_true',
                    help='whether normalize the observation')
parser.add_argument('--latent_dim', type=int, default=10,
                    help='the latent state dimension')
parser.add_argument('--ff_dim', type=int, default=512,
                    help='trafo ff dim')
parser.add_argument('--num_heads', type=int, default=16,
                    help='number of heads')
parser.add_argument('--add_data_first', action='store_true',
                    help='should use this for latent models')
parser.add_argument('--learning_rule', type=str, default='hebb',
                    choices=['hebb', 'oja', 'delta'], help='learning rule')
parser.add_argument('--adjoint', action='store_true',
                    help='whether use adjoint backprop')
parser.add_argument('--ode_tol', type=float, default=1e-3,
                    help='the relative error tolerance of ODE networks')
parser.add_argument('--ode_dim', type=int, default=5,
                    help='the number of hidden units in ODE network')
parser.add_argument('--enc_hidden_to_latent_dim', type=int, default=5,
                    help='the number of hidden units for hidden to latent')
parser.add_argument('--lr', type=float, default=9e-4,
                    help='the learning rate for training environment model')
parser.add_argument('--batch_size', type=int, default=32,
                    help='mini-batch size for training environment model')
parser.add_argument('--epochs', type=int, default=150,
                    help='number of epochs for training environment model')
parser.add_argument('--iters', type=int, default=12000,
                    help='number of iterations for training environment model')
parser.add_argument('--trajs', type=int, default=1000,
                    help='number of trajs for training environment model')
parser.add_argument('--eps_decay', type=float, default=1e-4,
                    help='the linear decay rate for scheduled sampling')
parser.add_argument('--max_steps', type=int,
                    help='max steps for running policy and traj. generation')
parser.add_argument('--episodes', type=int, default=1000,
                    help='the number of episodes for running policy')
parser.add_argument('--mem_size', type=int, default=int(1e5),
                    help='the size of experience replay buffer')
parser.add_argument('--log', action='store_true',
                    help='using logger or print')
parser.add_argument('--mpc_ac', action='store_true',
                    help='model predictive control for actor-critic')
parser.add_argument('--mb_epochs', type=int, default=10,
                    help='the epochs for iterative training')
parser.add_argument('--mf_epochs', type=int, default=240,
                    help='the epochs for iterative training')
parser.add_argument('--planning_horizon', type=int, default=15,
                    help='the planning horizon for environment model')
parser.add_argument('--env_steps', type=int, default=4000,
                    help='the number of environment steps per epoch')
parser.add_argument('--use_wandb', action='store_true', help='use wandb')
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--individual_dir', action='store_true',
                    help='dir struct dedicated to this job')
args = parser.parse_args()

if not os.path.exists("models/"):
    utils.makedirs("models/")
if not os.path.exists("logs/"):
    utils.makedirs("logs/")
if not os.path.exists("results/"):
    utils.makedirs("results/")

# seed for reproducibility
exp_id = int(random.SystemRandom().random() * 100000)

# to avoid restart
actual_seed = args.seed + args.num_restarts * 1000

random.seed(actual_seed)
np.random.seed(actual_seed)
torch.manual_seed(actual_seed)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(actual_seed)
    torch.cuda.manual_seed_all(actual_seed)


if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    if not os.path.exists(f"models/{exp_id}"):
        utils.makedirs(f"models/{exp_id}")

    wandb.init(dir=f'./models/{exp_id}',
               project=project_name,
               settings=wandb.Settings(start_method='fork'))
    # or `settings=wandb.Settings(start_method='thread')`
    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//{args.seed}//" \
                         f"{args.train_env_model}/{args.world_model}" \
                         f"{args.latent_policy}/{args.num_restarts}" \
                         f"{args.model}/{args.trained_model_path}/{args.env}" \
                         f"/{args.timer}/{args.gamma}/" \
                         f"{args.obs_normal}/{args.latent_dim}" \
                         f"/{args.ode_tol}" \
                         f"dim{args.ode_dim}/{args.enc_hidden_to_latent_dim}" \
                         f"lr{args.lr}/batch{args.batch_size}/{args.epochs}" \
                         f"/iter{args.iters}/{args.trajs}/{args.eps_decay}" \
                         f"/{args.max_steps}/{args.episodes}/{args.mem_size}" \
                         f"/{args.log}/{args.mpc_ac}/{args.mb_epochs}/" \
                         f"{args.mf_epochs}/{args.planning_horizon}/" \
                         f"{args.env_steps}/ff{args.ff_dim}/" \
                         f"head{args.num_heads}/add{args.add_data_first}" \
                         f"rule{args.learning_rule}/ad{args.adjoint}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.train_env_model=args.train_env_model
    config.world_model=args.world_model
    config.latent_policy=args.latent_policy
    config.num_restarts=args.num_restarts
    config.model=args.model
    config.trained_model_path=args.trained_model_path
    config.env=args.env
    config.timer=args.timer
    config.seed=args.seed
    config.gamma=args.gamma
    config.obs_normal=args.obs_normal
    config.latent_dim=args.latent_dim
    config.ode_tol=args.ode_tol
    config.ode_dim=args.ode_dim
    config.enc_hidden_to_latent_dim=args.enc_hidden_to_latent_dim
    config.lr=args.lr
    config.batch_size=args.batch_size
    config.epochs=args.epochs
    config.iters=args.iters
    config.trajs=args.trajs
    config.eps_decay=args.eps_decay
    config.max_steps=args.max_steps
    config.episodes=args.episodes
    config.mem_size=args.mem_size
    config.log=args.log
    config.mpc_ac=args.mpc_ac
    config.mb_epochs=args.mb_epochs
    config.mf_epochs=args.mf_epochs
    config.planning_horizon=args.planning_horizon
    config.env_steps=args.env_steps
    config.ff_dim=args.ff_dim
    config.num_heads=args.num_heads
    config.add_data_first=args.add_data_first
    config.learning_rule=args.learning_rule
    config.adjoint=args.adjoint
else:
    use_wandb = False

if args.env == 'grid':
    simulator = WindyGridSimulator()
elif args.env == 'acrobot':
    simulator = AcrobotSimulator()
elif args.env == 'hiv':
    simulator = HIVSimulator()
elif args.env == 'half_cheetah':
    simulator = HalfCheetahSimulator()
elif args.env == 'swimmer':
    simulator = SwimmerSimulator()
elif args.env == 'hopper':
    simulator = HopperSimulator()
else:
    raise NotImplementedError

simulator.seed(actual_seed)

if args.individual_dir:
    save_dir_name = f"seed{args.seed}"
else:
    save_dir_name = exp_id

if args.save_all:
    if not os.path.exists(f"models/{save_dir_name}"):
        utils.makedirs(f"models/{save_dir_name}")
    model_latest_ckpt_path = \
        f'models/{save_dir_name}/{args.model}_{args.env}.ckpt_model_latest'
    policy_latest_ckpt_path = \
        f'models/{save_dir_name}/{args.model}_{args.env}.ckpt_policy_latest'
    replay_mem_latest_path = \
        f'models/{save_dir_name}/{args.model}_{args.env}.ckpt_mem_latest'

    bk_model_latest_ckpt_path = \
        f'models/{save_dir_name}/{args.model}_{args.env}.ckpt_model_latest_bk'
    bk_policy_latest_ckpt_path = \
        f'models/{save_dir_name}/{args.model}_{args.env}.ckpt_policy_latest_bk'
    bk_replay_mem_latest_path = \
        f'models/{save_dir_name}/{args.model}_{args.env}.ckpt_mem_latest_bk'

# ckpt_path = 'models/{}_{}_{}.ckpt'.format(args.model, args.env, exp_id)
ckpt_path = (f'models/{args.model}_{args.env}_{exp_id}_{args.seed}_'
             f'{args.num_restarts}.ckpt')

if args.log:
    # log_path = 'logs/log_{}_{}_{}.log'.format(args.model, args.env, exp_id)
    log_path = (f'logs/log_{args.model}_{args.env}_{exp_id}_{args.seed}_'
                f'{args.num_restarts}.log')
    logger = utils.get_logger(
        logpath=log_path, filepath=os.path.abspath(__file__))
else:
    logger = None

utils.logout(logger,
             f'Experiment: {exp_id}, Model: {args.model}, '
             f'Environment: {args.env} {repr(simulator)}, Seed: {args.seed}, '
             f'Actual seed: {actual_seed}, Restart: {args.num_restarts}')

utils.logout(logger, f"torch version: {torch.__version__}")

utils.logout(logger, f"Command executed: {sys.argv[:]}")
utils.logout(logger, f"Args: {json.dumps(args.__dict__, indent=2)}")

utils.logout(logger,
             f'gamma: {args.gamma}, latent_dim: {args.latent_dim}, '
             f'lr: {args.lr}, batch_size: {args.batch_size}, '
             f'eps_decay: {args.eps_decay}, max steps: {args.max_steps}, '
             f'latent_policy: {args.latent_policy}, '
             f'obs_normal: {args.obs_normal}')

utils.logout(logger, '*' * 50)

oderl = MBRL(simulator,
             gamma=args.gamma,
             mem_size=args.mem_size,
             latent_dim=args.latent_dim,
             ff_dim=args.ff_dim,
             num_heads=args.num_heads,
             batch_size=args.batch_size,
             lr=args.lr,
             ode_tol=args.ode_tol,
             ode_dim=args.ode_dim,
             enc_hidden_to_latent_dim=args.enc_hidden_to_latent_dim,
             eps_decay=args.eps_decay,
             model=args.model,
             timer_type=args.timer,
             latent_policy=args.latent_policy,
             obs_normal=args.obs_normal,
             exp_id=exp_id,
             trained_model_path=args.trained_model_path,
             ckpt_path=ckpt_path,
             logger=logger,
             add_data_first=args.add_data_first,
             learning_rule=args.learning_rule,
             adjoint=args.adjoint,
             use_wandb=use_wandb)

# log model info
utils.logout(logger, oderl.model)
utils.logout(
    logger, f"Number of params (w/o policy): {oderl.model.num_params()}")


if args.train_env_model:
    utils.logout(logger, '*' * 10 + ' Collecting random rollouts ' + '*' * 10)
    for _ in range(args.trajs):
        oderl.run_policy(eps=1, max_steps=args.max_steps, store_trans=False,
                         store_traj=True, optimize_mf=False, cut_length=0,
                         val_ratio=0)
    for _ in range(args.trajs // 10):
        oderl.run_policy(eps=1, max_steps=args.max_steps, store_trans=False,
                         store_traj=True, optimize_mf=False, cut_length=0,
                         val_ratio=1)
    oderl.train_env_model(num_iters=args.iters)

if args.world_model:
    is_model_free = bool(args.model == 'free')
    choice = {True: oderl.run_policy, False: oderl.generate_traj_from_env_model}
    dic = {'rewards': [], 'trials': []}
    for i in range(args.episodes):
        t = time.time()
        choice[is_model_free](max_steps=args.max_steps)
        reward, _ = oderl.run_policy(eps=0.05, max_steps=args.max_steps,
                                     store_trans=False, store_traj=False,
                                     optimize_mf=False)
        dic['rewards'].append(reward)
        utils.logout(logger,
                     "Episode %d | rewards = %.6f | time = %.6f s" % (i + 1, dic['rewards'][-1], time.time() - t))
        if (i + 1) % 100 == 0:
            torch.save(dic, 'results_t/{}_{}_reward_{}.ckpt'.format(args.model, args.env, args.num_restarts))
    for _ in range(100):
        dic['trials'].append(
            oderl.run_policy(eps=0.05, max_steps=args.max_steps,
                             store_trans=False, store_traj=False,
                             optimize_mf=False)[0])
    utils.logout(logger, 'Average reward over last 100 trials: %f' % (sum(dic['trials'][-100:]) / 100))
    torch.save(dic, 'results/{}_{}_reward_{}.ckpt'.format(args.model, args.env, args.num_restarts))
    utils.logout(logger, '*' * 10 + ' Done ' + '*' * 10)

if args.mpc_ac:
    dic = {'rewards': [], 'trials': [], 'env_steps': []}
    total_env_steps = 0
    total_episodes = 0

    if args.restart_from != '':
        # load model
        old_model_path = \
            f'{args.restart_from}/{args.model}_{args.env}.ckpt_model_latest'
        assert os.path.exists(old_model_path), f"file: {old_model_path}"
        utils.logout(logger, f'Loading model params from {old_model_path}')
        ckpt = torch.load(old_model_path)
        oderl.model.load_state_dict(ckpt['model_state_dict'])
        oderl.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        total_episodes = ckpt['total_episodes']
        total_env_steps = ckpt['total_env_steps']

        # load policy
        old_model_path = \
            f'{args.restart_from}/{args.model}_{args.env}.ckpt_policy_latest'
        assert os.path.exists(old_model_path)
        utils.logout(logger, f'Loading policy params from {old_model_path}')
        ckpt = torch.load(old_model_path)
        oderl.policy.policy_actor.load_state_dict(ckpt['actor_state_dict'])
        oderl.policy.optimizer_actor.load_state_dict(
            ckpt['actor_optimizer_state_dict'])
        oderl.policy.policy_critic.load_state_dict(ckpt['critic_state_dict'])
        oderl.policy.optimizer_critic.load_state_dict(
            ckpt['critic_optimizer_state_dict'])

        # set epoch
        prev_epoch = ckpt['mb_epoch']
        utils.logout(logger, f'*** Start from epoch {prev_epoch} ***')
        # args.mb_epochs = args.mb_epochs - prev_epoch

        # reward etc
        old_model_path = \
            f'{args.restart_from}/reward_{args.model}_{args.env}.ckpt'
        assert os.path.exists(old_model_path)
        dic = torch.load(old_model_path)  # overwrite

        # replay buffer
        old_model_path = \
            f'{args.restart_from}/{args.model}_{args.env}.ckpt_mem_latest'
        assert os.path.exists(old_model_path)
        ckpt = torch.load(old_model_path)
        oderl.memory_traj_train.memory = ckpt['memory_traj_train_memory']
        oderl.memory_traj_train.position = ckpt['memory_traj_train_position']
        oderl.memory_traj_test.memory = ckpt['memory_traj_test_memory']
        oderl.memory_traj_test.position = ckpt['memory_traj_test_position']
        oderl.memory_trans.memory = ckpt['memory_tras_memory']
        oderl.memory_trans.position = ckpt['memory_tras_position']

        # running stats
        oderl.rms.n = ckpt['rms_n']
        oderl.rms.m = ckpt['rms_m']
        oderl.rms.s = ckpt['rms_s']

    else:
        prev_epoch = 0
        # random rollout
        rewards, steps, total_episodes, total_env_steps, eval_reward = \
            oderl.mbmf_rollout(
                'random', 3 * args.env_steps, args.max_steps, total_episodes,
                total_env_steps, cur_epoch=prev_epoch,
                store_trans=True, store_traj=True,
                val_ratio=0.1, planning_horizon=args.planning_horizon)
        dic['env_steps'].extend(steps)
        dic['rewards'].extend(rewards)

    for i in range(prev_epoch, max(args.mf_epochs, args.mb_epochs)):
        if i < args.mb_epochs:
            # model training
            utils.logout(
                logger,
                '*' * 10 + ' Training the environment model ' + '*' * 10)
            oderl.train_env_model_early_stopping(
                num_epochs=args.epochs, passes=max(15 - i, 3))

            # MBMF rollout
            rewards, steps, total_episodes, total_env_steps, eval_reward = \
                oderl.mbmf_rollout(
                    'mbmf', args.env_steps, args.max_steps, total_episodes,
                    total_env_steps, cur_epoch=i + 1,
                    store_trans=True, store_traj=True, val_ratio=0.1,
                    planning_horizon=args.planning_horizon)
            dic['env_steps'].extend(steps)
            dic['rewards'].extend(rewards)
            dic['trials'].append(eval_reward)

        # MF rollout (only used for model-free policy)
        if i < args.mf_epochs:
            rewards, steps, total_episodes, total_env_steps, eval_reward = \
                oderl.mbmf_rollout(
                    'mf', args.env_steps, args.max_steps, total_episodes,
                    total_env_steps, cur_epoch=i + 1,
                    store_trans=True, store_traj=True, val_ratio=0.1,
                    planning_horizon=args.planning_horizon)
            dic['env_steps'].extend(steps)
            dic['rewards'].extend(rewards)
            dic['trials'].append(eval_reward)

        if args.save_all:
            # back up old files os.rename(
            if os.path.exists(model_latest_ckpt_path):
                os.rename(model_latest_ckpt_path, bk_model_latest_ckpt_path)
            if os.path.exists(policy_latest_ckpt_path):
                os.rename(policy_latest_ckpt_path, bk_policy_latest_ckpt_path)
            if os.path.exists(replay_mem_latest_path):
                os.rename(replay_mem_latest_path, bk_replay_mem_latest_path)
            # save params
            torch.save({
                'total_env_steps': total_env_steps,
                'total_episodes': total_episodes,
                'model_state_dict': oderl.model.state_dict(),
                'optimizer_state_dict': oderl.optimizer.state_dict(),
                'mb_epoch': i + 1,
            }, model_latest_ckpt_path)

            torch.save({
                'actor_state_dict': oderl.policy.policy_actor.state_dict(),
                'actor_optimizer_state_dict': \
                    oderl.policy.optimizer_actor.state_dict(),
                'critic_state_dict': oderl.policy.policy_critic.state_dict(),
                'critic_optimizer_state_dict': \
                    oderl.policy.optimizer_critic.state_dict(),
                'mb_epoch': i + 1,
            }, policy_latest_ckpt_path)

            # save replay buffers and running stats
            torch.save({
                'memory_traj_train_memory': oderl.memory_traj_train.memory,
                'memory_traj_train_position': oderl.memory_traj_train.position,
                'memory_traj_test_memory': oderl.memory_traj_test.memory,
                'memory_traj_test_position': oderl.memory_traj_test.position,
                'memory_tras_memory': oderl.memory_trans.memory,
                'memory_tras_position': oderl.memory_trans.position,
                'rms_n': oderl.rms.n,
                'rms_m': oderl.rms.m,
                'rms_s': oderl.rms.s,
            }, replay_mem_latest_path)

            # save reward etc
            torch.save(
                dic,
                f'models/{save_dir_name}/reward_{args.model}_{args.env}.ckpt')

        torch.save(dic, 
                   f'results/{args.model}_{args.env}_seed_{args.seed}_'
                   f'restart_{args.num_restarts}_exp_{exp_id}.ckpt')

    utils.logout(logger, '*' * 10 + ' Done ' + '*' * 10)
