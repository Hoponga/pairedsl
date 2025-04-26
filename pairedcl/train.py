from agents.classifier_agent  import ClassifierAgent
from envs.task_generator      import TaskGenerator
from envs.classification_env  import ClassificationEnv, VecClassificationEnv
from runners.adversarial_runner import AdversarialRunner
from utils.arguments          import parser
from utils.make_agent         import make_agent 
import wandb, torch, random

args = parser.parse_args()
wandb.init(project='paired-classification', config=vars(args))

# 1) build dataset pool
train_pool, val_pool = data.datasets.get_pool(args.dataset_root)

# 2) instantiate generator, protagonist, antagonist
gen_policy   = TaskGenerator(args)
proto_agent  = ClassifierAgent(model=ResNet18(), optimizer='adam', args=args, num_envs=args.num_envs)
antag_agent  = ClassifierAgent(model=WideResNet(),  optimizer='adam', args=args, num_envs=args.num_envs)





# 3) wrap in VecEnvs so signatures match
ued_env   = VecClassificationEnv(TaskGenerator=gen_policy,    n_envs=args.num_envs)
task_env  = VecClassificationEnv(TaskPool=(train_pool,val_pool), n_envs=args.num_envs)

runner = AdversarialRunner(
            venv=task_env,
            ued_venv=ued_env,
            agent=proto_agent,
            adversary_agent=antag_agent,
            adversary_env=gen_policy,
            args=args)




for update in range(args.num_updates):
    stats = runner.run()
    wandb.log(stats, step=update)
    print(stats)