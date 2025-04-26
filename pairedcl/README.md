PAIRED algorithm for continual learning -- project organization: 

pairedcl/                      # new project root
│
├── train.py                   # main driver (mirrors original train.py)
├── evaluate.py                # optional offline eval script
│
├── agents/
│   ├── classifier_agent.py    # protagonist / antagonist
│   └── baseline_agent.py      # frozen ResNet or ensemble (optional)
│
├── envs/
│   ├── task_spec.py           # TaskSpec dataclass
│   ├── task_generator.py      # π_E  (environment-generator policy)
│   ├── classification_env.py  # gym.Env wrapper around a TaskSpec
│   └── vec_env.py             # Vectorised & parallel wrapper (like Baselines’)
│
├── runners/
│   └── adversarial_runner.py  # classification-flavoured AdversarialRunner
│
├── algorithms/
│   ├── ppo.py                 # generic PPO / A2C (re-used by every agent)
│   └── storage.py             # RolloutStorage adapted for classification
│
├── data/
│   └── datasets.py            # utilities that materialise DataLoaders
│
└── utils/
    ├── arguments.py           # argparse definitions
    ├── metrics.py             # accuracy, n-shot eval, etc.
    └── logger.py              # console + WandB