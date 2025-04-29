# train_multi_task.py ────────────────────────────────────────────────
import random, argparse, collections, itertools
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

from pairedcl.envs.task_spec          import TaskSpec
from pairedcl.envs.task_generator     import TaskGenerator
from pairedcl.envs.classification_env import ClassificationEnv
from pairedcl.utils.evaluator                         import Evaluator          # class from earlier
from pairedcl.utils.make_agent                        import make_task_generator

# We use the exact same MLP as in PAIRED-CL
class MLP(nn.Module):
    def __init__(self, in_dim=28*28, hidden=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, in_dim), nn.ReLU(),
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

# ---------- replay buffer ------------------------------------------
Sample = collections.namedtuple("Sample", "x y")

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.cap = capacity
        self.mem = collections.deque(maxlen=capacity)
    def add_batch(self, xs, ys):
        for x, y in zip(xs, ys):
            self.mem.append(Sample(x.cpu(), y.cpu()))
    def sample(self, k):
        idx = np.random.choice(len(self.mem), size=k, replace=False)
        xs, ys = zip(*(self.mem[i] for i in idx))
        return torch.stack(xs), torch.tensor(ys)

# -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T",              type=int,   default=2, help="# tasks")
    p.add_argument("--batch",          type=int,   default=64)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--replay_ratio",   type=float, default=0.5,
                   help="fraction of each step drawn from replay buffer")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # Define the env policy (that will never be trained, just as the wrapper for dataset transformations) 
    tg = make_task_generator(device) 


    
    train_specs = [tg.action_to_taskspec(torch.rand(tg.action_dim)*2 - 1)
                   for _ in range(args.T)]
    dummy_spec  = TaskSpec("mnist-train", transforms=[])        # bootstrap

    class_env = ClassificationEnv(dummy_spec, batch_size=args.batch, device=device)
    evaluator = Evaluator(tg, class_env, T=args.T, batches_per_task=8, device=device)

    # 3) model, optim ---------------------------------------------------------
    model = MLP().to(device)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    ce    = nn.CrossEntropyLoss()


    # 4) replay buffer --------------------------------------------------------
    buffer = ReplayBuffer(capacity=20_000)

    # 5) training loop --------------------------------------------------------
    for epoch in range(1, args.epochs+1):
        random.shuffle(train_specs)
        running_loss = 0.0
        running_acc  = 0.0
        n_batches    = 0

        for spec in train_specs:
            # switch env to this task
            _, _ = class_env.reset(task_spec=spec)
            loader_iter = iter(class_env.loader)

            for imgs, labels in loader_iter:
                imgs, labels = imgs.to(device), labels.to(device)

                # store current task batch in replay buffer
                buffer.add_batch(imgs, labels)

                # ----- build mixed batch ------------------------------------
                if len(buffer.mem) > 0 and args.replay_ratio > 0:
                    k_replay = int(args.batch * args.replay_ratio)
                    k_fresh  = args.batch - k_replay
                    # truncate fresh batch if needed
                    imgs_f, labels_f = imgs[:k_fresh], labels[:k_fresh]
                    imgs_r, labels_r = buffer.sample(k_replay)
                    imgs_r, labels_r = imgs_r.to(device), labels_r.to(device)
                    imgs   = torch.cat([imgs_f, imgs_r], dim=0)
                    labels = torch.cat([labels_f, labels_r], dim=0)

                # ----- gradient step ----------------------------------------
                logits = model(imgs)
                loss   = ce(logits, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                running_loss += loss.item()
                running_acc  += (logits.argmax(1) == labels).float().mean().item()
                n_batches    += 1

        # 6) stats ------------------------------------------------------------
        train_loss = running_loss / n_batches
        train_acc  = running_acc  / n_batches
        eval_acc   = evaluator.evaluate(model)

        print(f"[epoch {epoch:03d}]   "
              f"train_acc={train_acc:.3f}   "
              f"eval_acc={eval_acc:.3f}   "
              f"loss={train_loss:.4f}")

if __name__ == "__main__":
    main()
