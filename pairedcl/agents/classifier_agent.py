


class ClassifierAgent:
    def __init__(self, model, optimizer, args, num_envs): ...
    # === RL-style API ===
    def act(self, obs, hidden, mask):
        """
        obs : batch of images
        return value est., discrete action (= predicted class index),
               log-dist (Softmax logits), new hidden.
        """
    def process_action(self, action_cpu): ...  # identity for classification
    def update(self) -> Tuple[val_loss, policy_loss, entropy, info]: ...