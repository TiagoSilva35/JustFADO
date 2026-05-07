NSGA2_PREQ_CONFIG_PATH = 'files/nsga2_prequential_config.yaml'
NSGA2_TREE_CONFIG_PATH = 'files/nsga2_tree_config.yaml'
DEFAULT_SEED_RUNS = 1
DEFAULT_RANDOM_SEED_MIN = 1
DEFAULT_RANDOM_SEED_MAX = 2_147_483_647
RFR_CONFIG = {
    'approach': 'rfr',
    'backbone': 'netregression',
    'hidden_dim': 50,
    'n_ensemble': 4,
    'learning_rate': 1e-3,
    'rho': 1e-4,
    'penalty_coefficient': 1.0,
    'fcr_threshold': 0.8,
    'train_batch_size': 1,
    'buffer_size': 512,
    'adv_hidden_dim': 32,
}
OUTPUT_DIR = 'files/experiments'
