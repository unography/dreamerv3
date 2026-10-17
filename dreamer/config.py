"""The hyperparameters of the agent.

The defaults are the `dmc_proprio` settings of the reference implementation.
It uses the smallest model size of DreamerV3 (`size1m` in the reference) for
DMC tasks with proprioceptive observations.

Reference: appendix "Hyperparameters" of the DreamerV3 paper, and
`dreamerv3/configs.yaml` of the reference implementation.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    # The model size.
    deter: int = 512  # The size of the deterministic state h.
    hidden: int = 64  # The width of the hidden layers of the RSSM.
    stoch: int = 32  # The number of categorical variables in z.
    classes: int = 4  # The number of classes of each categorical variable.
    blocks: int = 8  # The number of blocks of the recurrent network.
    units: int = 64  # The width of the encoder, decoder, and heads.

    # The world model.
    replay_context: bool = True  # Start sequences from the stored RSSM state.
    unimix: float = 0.01
    free_nats: float = 1.0
    horizon: int = 333  # The discount factor is 1 - 1 / horizon.

    # The actor and the critic.
    imag_horizon: int = 15
    lam: float = 0.95
    entropy_scale: float = 3e-4
    min_std: float = 0.1
    max_std: float = 1.0
    slow_reg: float = 1.0
    slow_rate: float = 0.02
    return_norm_rate: float = 0.01

    # The scales of the losses.
    dyn_scale: float = 1.0
    rep_scale: float = 0.1
    rec_scale: float = 1.0
    rew_scale: float = 1.0
    con_scale: float = 1.0
    policy_scale: float = 1.0
    value_scale: float = 1.0
    repval_scale: float = 0.3

    # The optimizer.
    lr: float = 4e-5
    agc: float = 0.3
    eps: float = 1e-20
    beta1: float = 0.9
    beta2: float = 0.999
    warmup: int = 1000
