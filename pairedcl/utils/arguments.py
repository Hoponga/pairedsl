# arguments.py – command‑line interface for PAIRED‑classification
"""Parse and post‑process all command‑line arguments used across the
PAIRED‑for‑classification code‑base.

Typical usage (inside *train.py*):

    from utils.arguments import get_args
    args = get_args()
    print(f"Running on device {args.device}…")

All flags are grouped in thematic sections so you can quickly see what
controls what.  Every flag has a sensible default so you can run the demo
with:

    python train.py --dataset cifar100  # or just: python train.py
"""
from __future__ import annotations

import argparse
import pathlib
import typing as _t
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
try:
    import torch  # optional ‑ only used for automatic CUDA detection
except ImportError:  # pragma: no cover – torch not installed in docs
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    """Return a fully‑initialised :class:`argparse.ArgumentParser`."""

    parser = argparse.ArgumentParser(
        "PAIRED‑classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------- dataset / task ------------------------------------------------
    g = parser.add_argument_group("Task / dataset")
    g.add_argument("--dataset", type=str, default="cifar100",
                   choices=["cifar10", "cifar100", "imagenet", "flowers"],
                   help="name of the dataset pool to sample tasks from")
    g.add_argument("--dataset-root", type=str, default="~/data",
                   help="root folder that contains the raw datasets")
    g.add_argument("--n-classes-per-task", type=int, default=5,
                   help="how many distinct classes go into each sampled task")
    g.add_argument("--shots", type=int, default=20,
                   help="# training samples per class given to the learner")

    # -------- generator rollout --------------------------------------------
    g = parser.add_argument_group("Environment‑generator policy (π_E)")
    g.add_argument("--env-steps", type=int, default=15,
                   help="# placement decisions per task episode")
    g.add_argument("--env-lr", type=float, default=1e-4,
                   help="learning‑rate for π_E optimiser")
    g.add_argument("--regret-alpha", type=float, default=1.0,
                   help="scale factor applied to regret when fed as reward")
    g.add_argument("--replay-size", type=int, default=100,
                   help="buffer size if using Level Replay / ALR")

    # -------- learner rollout ----------------------------------------------
    g = parser.add_argument_group("Protagonist / antagonist agents")
    g.add_argument("--agent-steps", type=int, default=10,
                   help="# batches (gradient steps) per task episode")
    g.add_argument("--lr", type=float, default=3e-4,
                   help="learning‑rate for protagonist / antagonist models")
    g.add_argument('--adv_num_mini_batch',type=int,default=1,help='number of batches for PPO used by adversary')
    g.add_argument('--adv_ppo_epoch',type=int,default=5,help='Number of PPO epochs used by adversary')
    g.add_argument(
    '--num_processes',
    type=int,
    default=1,
    help='How many training CPU processes to use for rollouts')
    g.add_argument(
    '--num_steps',
    type=int,
    default=256,
    help='Number of rollout steps for PPO')

    g.add_argument('--context_obs_shape', type = int, default = 3, help = 'Number of scalar features that the adversary environment takes in as context to making a new task')
    # -------- PPO / RL hyper‑params ----------------------------------------
    g = parser.add_argument_group("PPO / advantage settings")
    g.add_argument("--gamma", type=float, default=0.99)
    g.add_argument("--gae-lambda", type=float, default=0.95)
    g.add_argument("--clip-eps", type=float, default=0.2)
    g.add_argument("--entropy-coef", type=float, default=0.01)
    g.add_argument("--value-loss-coef", type=float, default=0.5)
    g.add_argument("--max-grad-norm", type=float, default=0.5)


    g.add_argument(
    '--clip_param',
    type=float,
    default=0.2,
    help='PPO clip parameter')
    g.add_argument(
    '--clip_value_loss',
    type=str2bool,
    default=True,
    help='PPO clip value loss')
    g.add_argument(
    '--adv_clip_reward',
    type=float,
    default=None,
    help='Clip adversary rewards')
    g.add_argument(
    '--eps',
    type=float,
    default=1e-5,
    help='RMSprop optimizer epsilon')
    g.add_argument(
    '--alpha',
    type=float,
    default=0.99,
    help='RMSprop optimizer apha')

    # -------- training loop -------------------------------------------------
    g = parser.add_argument_group("Outer loop")
    g.add_argument("--num-updates", type=int, default=20_000,
                   help="total generator/protagonist updates")
    g.add_argument("--num-envs", type=int, default=8,
                   help="# parallel environments / tasks")

    g.add_argument(
    '--log_grad_norm',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log gradient norms")
    g.add_argument(
    '--log_action_complexity',
    type=str2bool, nargs='?', const=True, default=False,
    help="Log action trajectory complexity metric")
    g.add_argument(
    '--test_interval',
    type=int,
    default=250,
    help='Evaluate on test envs every n updates.')

    g.add_argument("--k_inner", type=int, default=4,
                   help="batches to train classifier per task")
    g.add_argument("--antagonist_delta", type = int, default = 4, 
                   help="Extra number of batches to train antagonist per task")
    g.add_argument("--rollout", type=int, default=16,
                   help="tasks per PPO update")
    g.add_argument("--epochs", type=int, default=10000)

    # -------- logging / I/O -------------------------------------------------
    g = parser.add_argument_group("Logging & checkpoints")
    g.add_argument("--log-dir", type=str, default="./logs")
    g.add_argument("--save-interval", type=int, default=1_000)
    g.add_argument("--eval-interval", type=int, default=500)
    g.add_argument("--wandb-project", type=str, default="paired-classification")
    g.add_argument("--no-wandb", action="store_true",
                   help="disable Weights & Biases tracking")
    g.add_argument("--render", action="store_true",
                   help="visualise first task in each batch for debugging")

    # -------- misc ----------------------------------------------------------
    g = parser.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--cuda", action="store_true",
                   help="use CUDA if available")
    g.add_argument("--device", type=str, default=None,
                   help="explicit device override, e.g. 'cuda:1' or 'cpu'")

    return parser


def get_args(argv: _t.Sequence[str] | None = None) -> argparse.Namespace:
    """Parse *argv* (or ``sys.argv``) and perform light post‑processing."""

    parser = get_parser()
    args = parser.parse_args(argv)

    # --- canonicalise paths -------------------------------------------------
    args.dataset_root = pathlib.Path(args.dataset_root).expanduser().resolve()
    args.log_dir      = pathlib.Path(args.log_dir).expanduser().resolve()
    args.log_dir.mkdir(parents=True, exist_ok=True)

    # --- device handling ----------------------------------------------------
    if args.device is None:
        cuda_ok = torch and torch.cuda.is_available() and args.cuda
        args.device = "cuda" if cuda_ok else "cpu"

    return args


# ---------------------------------------------------------------------------
# CLI helper ----------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    _args = get_args()
    pprint.pprint(vars(_args))
