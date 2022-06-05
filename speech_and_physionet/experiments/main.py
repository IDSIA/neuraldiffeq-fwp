# main file to be executed

import os
import sys
import json
import time
from datetime import datetime
import argparse
import logging
import numpy as np
import random
import hashlib
import torch

import sepsis
import speech_commands
import uea


parser = argparse.ArgumentParser(
    description='Experiments with neural differential equations for weight '
                'update learning rules.')
parser.add_argument('--name_task', type=str, default='speech_commands',
                    choices=['speech_commands', 'sepsis', 'uea'])
parser.add_argument('--work_dir', default='save_logs', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--model_name', type=str, default='mygruode')
parser.add_argument('--seed', default=1, type=int, help='Seed.')

parser.add_argument('--hidden_channels', default=128, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--hidden_hidden_channels', default=40, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--num_hidden_layers', default=4, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--num_heads', default=8, type=int,
                    help='Transformer number of heads.')
parser.add_argument('--trafo_ff_dim', default=None, type=int,
                    help='Transformer ff dim to hidden dim ratio.')
parser.add_argument('--cde_use_v_laynorm', action='store_true',
                    help='use layer norm for v pre-projection')
parser.add_argument('--delta_ode_post_tahn', action='store_true',
                    help='put tahn after retrieval in delta ODE')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout rate.')

# training hyper-parameters:
# default per task:
# speech: max_epochs=200, batch_size=1024, base_lr=0.00005, grad_scale=100,
# sepsis: max_epochs=200, batch_size=1024, base_lr=0.0001, grad_scale=100
# uea, character: max_epochs=1000, batch_size=32, base_lr=0.001, grad_scale=1
parser.add_argument('--use_default_setting', action='store_true',
                    help='overwrite all training params by Kidger et al.s')
parser.add_argument('--max_epochs', default=200, type=int,
                    help='max number of training epochs. '
                         'We follow Kidger et al.s stopping criteria.')
parser.add_argument('--batch_size', default=1024, type=int,
                    help='batch size.')
parser.add_argument('--base_lr', default=5e-5, type=float,
                    help='initial learning rate.')
parser.add_argument('--grad_scale', default=100, type=float,
                    help='initial learning rate.')
parser.add_argument('--epoch_per_metric', default=10, type=int,
                    help='Report log every this epoch.')
# we never used this for fair comparisons, but it might help. we did not try.
parser.add_argument('--val_based_stopping', action='store_true',
                    help='val_based_stopping (train based by default.')
# we never used this for fair comparisons, but it might help. we did not try.
parser.add_argument('--auroc_based_stopping', action='store_true',
                    help='train_based_stopping.')

# task specific
parser.add_argument('--use_sepsis_intensity', action='store_true',
                    help='set intensity True.')
parser.add_argument('--uea_dataset_name', type=str, 
                    default='CharacterTrajectories')
parser.add_argument('--bce_pos_weight', default=10, type=float,
                    help='weight for positives for Binary CE.')
parser.add_argument('--uea_missing_rate', default=0.0, type=float,
                    help='uea missing rate.')

# for wandb
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

model_name = args.model_name

exp_str = ''
for arg_key in vars(args):
    if arg_key != 'seed':
        exp_str += str(getattr(args, arg_key)) + '-'

# taken from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
exp_hash = str(int(hashlib.sha1(exp_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8))

# Set work directory
args.work_dir = os.path.join(
    args.work_dir, args.name_task, model_name, exp_hash,
    f"{str(args.seed)}-{exp_str}" + time.strftime('%Y%m%d-%H%M%S'))

if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

work_dir_key = '/'.join(os.path.abspath(args.work_dir).split('/')[-3:])

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
loginf(f"Work dir: {args.work_dir}")

# wandb settings
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(
        project=project_name, settings=wandb.Settings(start_method='fork'))
    # or `settings=wandb.Settings(start_method='thread')`
    if args.job_name is None:
        wandb.run.name = f"{os.uname()[1]}//" \
                         f"{model_name}-{args.name_task}//" \
                         f"seed{args.seed}//" \
                         f"vnorm{args.cde_use_v_laynorm}/" \
                         f"post_tanh{args.delta_ode_post_tahn}" \
                         f"default-{args.use_default_setting}/" \
                         f"h{args.hidden_channels}/" \
                         f"hh{args.hidden_hidden_channels}/" \
                         f"L{args.num_hidden_layers}" \
                         f"n{args.num_heads}/ff{args.trafo_ff_dim}/" \
                         f"maxep{args.max_epochs}/b{args.batch_size}/" \
                         f"lr{args.base_lr}/gs{args.grad_scale}" \
                         f"val{args.val_based_stopping}/" \
                         f"auc{args.auroc_based_stopping}/" \
                         f"I{args.use_sepsis_intensity}/" \
                         f"miss{args.uea_missing_rate}/drop{args.dropout}" \
                         f"//PATH'{work_dir_key}'//"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.seed = args.seed
    config.name_task = args.name_task
    config.work_dir = args.work_dir
    config.model_name = args.model_name
    config.hidden_channels = args.hidden_channels
    config.hidden_hidden_channels = args.hidden_hidden_channels
    config.num_hidden_layers = args.num_hidden_layers
    config.num_heads = args.num_heads
    config.trafo_ff_dim = args.trafo_ff_dim
    config.cde_use_v_laynorm = args.cde_use_v_laynorm
    config.delta_ode_post_tahn = args.delta_ode_post_tahn
    config.use_default_setting = args.use_default_setting
    config.max_epochs = args.max_epochs
    config.batch_size = args.batch_size
    config.base_lr = args.base_lr
    config.grad_scale = args.grad_scale
    config.epoch_per_metric = args.epoch_per_metric
    config.val_based_stopping = args.val_based_stopping
    config.auroc_based_stopping = args.auroc_based_stopping
    config.use_sepsis_intensity = args.use_sepsis_intensity
    config.uea_missing_rate = args.uea_missing_rate
    config.dropout=args.dropout
else:
    use_wandb = False
# end wandb

# save args
loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

with open(f'{args.work_dir}/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

# set seed
loginf(f"Seed: {args.seed}")
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# set device
device = 'cuda'

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# set dataset
loginf(f"Dataset/Task: {args.name_task}")
if args.name_task == 'uea':
    loginf(f"UEA: {args.uea_dataset_name}")
loginf(f"Start time: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}.")

if args.name_task == 'speech_commands':
    if args.use_default_setting:
        loginf("== Using default hyper-params, ignoring args == ")
        results = speech_commands.main(
            model_name=args.model_name,
            hidden_channels=args.hidden_channels,
            hidden_hidden_channels=args.hidden_hidden_channels,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            trafo_ff_dim=args.trafo_ff_dim,
            dropout=args.dropout,
            cde_use_v_laynorm=args.cde_use_v_laynorm,
            delta_ode_post_tahn=args.delta_ode_post_tahn,
            device=device,
            use_wandb=args.use_wandb,
            loginf=loginf,
            dry_run=False)
    else:
        results = speech_commands.main(
            model_name=args.model_name,
            hidden_channels=args.hidden_channels,
            hidden_hidden_channels=args.hidden_hidden_channels,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            trafo_ff_dim=args.trafo_ff_dim,
            dropout=args.dropout,
            cde_use_v_laynorm=args.cde_use_v_laynorm,
            delta_ode_post_tahn=args.delta_ode_post_tahn,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            grad_scale=args.grad_scale,
            use_wandb=args.use_wandb,
            epoch_per_metric=args.epoch_per_metric,
            val_based_stopping=args.val_based_stopping,
            auroc_based_stopping=args.auroc_based_stopping,
            loginf=loginf,
            dry_run=False)
elif args.name_task == 'sepsis':
    if args.use_default_setting:
        loginf("== Using default hyper-params, ignoring args == ")
        results = sepsis.main(
            intensity=args.use_sepsis_intensity,
            model_name=args.model_name,
            hidden_channels=args.hidden_channels,
            hidden_hidden_channels=args.hidden_hidden_channels,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            trafo_ff_dim=args.trafo_ff_dim,
            dropout=args.dropout,
            cde_use_v_laynorm=args.cde_use_v_laynorm,
            delta_ode_post_tahn=args.delta_ode_post_tahn,
            device=device,
            use_wandb=args.use_wandb,
            loginf=loginf,
            dry_run=False)
    else:
        results = sepsis.main(
            intensity=args.use_sepsis_intensity,
            model_name=args.model_name,
            hidden_channels=args.hidden_channels,
            hidden_hidden_channels=args.hidden_hidden_channels,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            trafo_ff_dim=args.trafo_ff_dim,
            dropout=args.dropout,
            cde_use_v_laynorm=args.cde_use_v_laynorm,
            delta_ode_post_tahn=args.delta_ode_post_tahn,
            device=device,
            max_epochs=args.max_epochs,
            pos_weight=args.bce_pos_weight,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            grad_scale=args.grad_scale,
            use_wandb=args.use_wandb,
            epoch_per_metric=args.epoch_per_metric,
            val_based_stopping=args.val_based_stopping,
            auroc_based_stopping=args.auroc_based_stopping,
            loginf=loginf,
            dry_run=False)
elif args.name_task == 'uea':
    if args.use_default_setting:
        loginf("== Using default hyper-params, ignoring args == ")
        results = uea.main(
            dataset_name=args.uea_dataset_name,
            model_name=args.model_name,
            hidden_channels=args.hidden_channels,
            hidden_hidden_channels=args.hidden_hidden_channels,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            trafo_ff_dim=args.trafo_ff_dim,
            dropout=args.dropout,
            cde_use_v_laynorm=args.cde_use_v_laynorm,
            delta_ode_post_tahn=args.delta_ode_post_tahn,
            missing_rate=args.uea_missing_rate,
            device=device,
            use_wandb=args.use_wandb,
            loginf=loginf,
            dry_run=False)
    else:
        results = uea.main(
            dataset_name=args.uea_dataset_name,
            model_name=args.model_name,
            hidden_channels=args.hidden_channels,
            hidden_hidden_channels=args.hidden_hidden_channels,
            num_hidden_layers=args.num_hidden_layers,
            num_heads=args.num_heads,
            trafo_ff_dim=args.trafo_ff_dim,
            dropout=args.dropout,
            cde_use_v_laynorm=args.cde_use_v_laynorm,
            delta_ode_post_tahn=args.delta_ode_post_tahn,
            missing_rate=args.uea_missing_rate,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            base_lr=args.base_lr,
            grad_scale=args.grad_scale,
            use_wandb=args.use_wandb,
            epoch_per_metric=args.epoch_per_metric,
            val_based_stopping=args.val_based_stopping,
            auroc_based_stopping=args.auroc_based_stopping,
            loginf=loginf,
            dry_run=False)
else:
    assert False, f'Unknown task: {args.name_dataset}'

if args.name_task in ['speech_commands', 'uea']:
    log_str = (
        f'Train accuracy: {results.train_metrics.accuracy:.3f} '
        f'Val accuracy: {results.val_metrics.accuracy:.3f} '
        '\n'
        f'Test accuracy: {results.test_metrics.accuracy:.3f} ')
else:
    log_str = (
        f'Train accuracy: {results.train_metrics.accuracy:.3f} '
        f'Train AUC: {results.train_metrics.auroc:.3f} '
        f'Train precision: {results.train_metrics.average_precision:.3f} '
        f'Val accuracy: {results.val_metrics.accuracy:.3f} '
        f'Val AUC: {results.val_metrics.auroc:.3f} '
        f'Val precision: {results.val_metrics.average_precision:.3f}'
        '\n'
        f'Test accuracy: {results.test_metrics.accuracy:.3f} '
        f'Test AUC: {results.test_metrics.auroc:.3f} '
        f'Test precision: {results.test_metrics.average_precision:.3f}')

loginf(f"{log_str}")

loginf(f"End: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}.")
