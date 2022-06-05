import sys
import os
import time
import numpy as np
import random
from datetime import datetime
import logging
import json
import argparse
import hashlib

import signatory
import torch
from torchdiffeq import odeint, odeint_adjoint

from ingredients.trainer import train
from ingredients.prepare_data import ready_all_data_and_model
from utils.functions import save_pickle


# CLI's for paralellisation
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='Folder that holds the data.', default='UEA')
parser.add_argument('-ds', '--dataset', help='The name of the dataset to run.', default='EigenWorms')
parser.add_argument('-save_dir', '--save_dir', help='The total number of GPUs', default='./save_logs', type=str)

parser.add_argument('--report_every', help='Report log every this epochs',
                    default=5, type=int)
parser.add_argument('--final_layer_scaling', help='Final layer scaling',
                    default=10, type=int)
parser.add_argument('--seed', help='seed', default=1, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr', default=None)

parser.add_argument('--model_type', default='nrde', type=str)
parser.add_argument('--depth', help='depth of log signature', default=2, type=int)
parser.add_argument('--step', help='downsampling steps', default=4, type=int)
parser.add_argument('--hidden_dim', help='hidden dim', default=32, type=int)
parser.add_argument('--hidden_hidden_multiplier', help='hidden_hidden_multiplier for vector field', default=2, type=int)
parser.add_argument('--num_layers', help='num_layers for vector field', default=3, type=int)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--residual', action='store_true',
                    help='use residual connection for multi-layer model')

parser.add_argument('--scale_query_net', action='store_true')
parser.add_argument('--scale_ff', action='store_true')
parser.add_argument('--scale_out_proj', action='store_true')

parser.add_argument('--learning_rule', default='oja')
parser.add_argument('--num_heads', default=16, type=int)
parser.add_argument('--trafo_ff_dim', help='if None x4', default=None)

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

exp_str = ''
for arg_key in vars(args):
    if arg_key != 'seed':
        exp_str += str(getattr(args, arg_key)) + '-'

# taken from https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
exp_hash = str(int(hashlib.sha1(exp_str.encode("utf-8")).hexdigest(), 16) % (10 ** 8))

# Set work directory
args.save_dir = os.path.join(
    args.save_dir, args.model_type, args.learning_rule, exp_hash,
    f"{args.seed}-{time.strftime('%Y%m%d-%H%M%S')}")
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

log_file_name = f"{args.save_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
loginf(f"Save dir: {args.save_dir}")

loginf(f"Command executed: {sys.argv[:]}")
loginf(f"Args: {json.dumps(args.__dict__, indent=2)}")

model_type = args.model_type
depth = args.depth
step = args.step
hidden_dim = args.hidden_dim
hidden_hidden_multiplier = args.hidden_hidden_multiplier
num_layers = args.num_layers
adjoint = args.adjoint
num_heads = args.num_heads
trafo_ff_dim = args.trafo_ff_dim
learning_rule = args.learning_rule
tune_params = False

scale_query_net = args.scale_query_net
scale_ff = args.scale_ff
scale_out_proj = args.scale_out_proj

residual = args.residual

lr = float(args.lr) if args.lr is not None else None
batch_size=args.batch_size

trafo_ff_dim = int(trafo_ff_dim) if trafo_ff_dim is not None else None

model, train_dl, val_dl, test_dl = ready_all_data_and_model(
    model_type, depth, step, hidden_dim, hidden_hidden_multiplier,
    num_layers, tune_params, solver='rk4', adjoint=adjoint,
    trafo_ff_dim=trafo_ff_dim, num_heads=num_heads,
    batch_size=batch_size,
    scale_query_net=scale_query_net, scale_ff=scale_ff,
    scale_out_proj=scale_out_proj, residual=residual,
    learning_rule=learning_rule, ignore_model=False)

loginf(model)
loginf(f'Num params: {model.num_params()}')

model = model.to('cuda')

# Train model
model, results, history = train(
    model, train_dl, val_dl, test_dl,
    loss_str='ce',
    optimizer_name='adam',
    lr=lr,  # 0.032 / batch size
    max_epochs=1000,
    metrics=['loss', 'acc'],
    val_metric_to_monitor='acc',
    print_freq=args.report_every,
    epoch_per_metric=1,
    plateau_patience=15,
    plateau_terminate=60,
    gpu_if_available=True,
    gpu_idx=0,
    final_layer_scaling=args.final_layer_scaling,
    custom_metrics=None,
    loginf=loginf,
    save_dir=args.save_dir
)

# Save best values to metrics and history to file
for name, value in results.items():
    loginf(f'{name}, {value}')
save_pickle(history, args.save_dir + '/validation_history.pkl')
