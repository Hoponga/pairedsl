class TaskSpec(NamedTuple):
    id: str                  # hashable key
    meta: Dict               # any metadata
    train_loader: DataLoader
    val_loader: DataLoader

class TaskGenerator(nn.Module):          # π_E
    def reset(self): ...
    def forward(self, obs) -> Categorical: ...
    def step(self, action) -> (obs, 0., done, {})
    def finalize(self) -> TaskSpec        # when done==True

class Learner(nn.Module):                # π_P or π_B
    def forward(self, batch): ...
    def train_on_task(self, taskspec, n_epochs) -> float   # return accuracy
    def eval_on_task(self, taskspec)  -> float