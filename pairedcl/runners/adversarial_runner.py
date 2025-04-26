class AdversarialRunner(object): 

    def __init__(): 
        
    def run(self):
        # 1. sample TaskSpec from π_E
        obs_e, task_spec, done, _ = self.ued_venv.step(self.adversary_env.act())
        task_loader = materialise_loader(task_spec)

        # 2. protagonist trains
        for _ in range(self.args.k_inner_updates):
            imgs, labels = next(task_loader)
            logits = self.agent(imgs)
            loss = F.cross_entropy(logits, labels)
            self.agent.update(loss)

        # 3. antagonist evals (no grads)
        with torch.no_grad():
            acc_ant = accuracy(self.adversary_agent, task_loader)
            acc_pro = accuracy(self.agent, task_loader)

        # 4. compute reward & store in RolloutStorage
        reward_e = torch.clamp(acc_ant - acc_pro, 0., 1.)
        self.storage.insert(obs_e, act_e, reward_e, ...)

        # 5. step PPO for π_E (and antagonist soft-update τ)
        if self.storage.ready():
            self.ppo.update(self.storage)
        polyak_update(self.agent, self.adversary_agent, tau=self.args.polyak)
        return {"acc_pro": acc_pro, "acc_ant": acc_ant, "rew_e": reward_e.mean().item()}