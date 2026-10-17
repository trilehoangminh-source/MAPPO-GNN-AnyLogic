
from alpyne.data.spaces import Observation
from ..envs.matrix import Matrix

# ===== CAU HINH MO PHONG =====

FLEETSIZE = 2

env = Matrix(
    model_path=None,
    fleetsize=FLEETSIZE,
    max_fleetsize=20,

    # Su dung Behavior co san cua Tobias
    pseudo_routing=True,
    pseudo_dispatcher=True,
    pseudo_dispatcher_clever=True,
    pseudo_dispatcher_rule="shortest_quene",

    sim_config={
        "dispatch": True,
        "routing_ma": True,
        "dispatching_ma": True,

        "routing_interval": 2,
        "dispatching_interval": 120,
        "block_timeout": 120,

        "fixed_fleets": False,
        "remove_all_blocked": False,

        "availability": 0.9,
        "mttr": 5,
        "io_quote": 0.9,
    },
)

env_config = env.config


# ===== GUI CAU HINH CHO ANYLOGIC =====

def get_config(name: str):

    # Tuong thich ten tham so trong Main cua Tobias
    if name == "obsrout_include_part_info":
        name = "obs_include_part_info"

    if name in env_config.names():
        return env_config.values()[
            env_config.names().index(name)
        ]

    raise KeyError(f"Unknown configuration: {name}")


# ===== PYTHON TINH ACTION CHO AGV =====

def get_action(
    observation,
    caller,
    n_nodes,
    context,
    statTitles,
    statValues,
    time
):
    
    print("PYTHON DANG TINH ACTION - TIME:", time, flush=True)
    with open(r"C:\MARL_AnyLogic\pypeline_debug.txt", "a") as f:
        f.write(f"Python get_action called at time = {time}\n")

    alpyne_obs = Observation(
        obs=observation,
        caller=caller,
        n_nodes=n_nodes,
        networkcontext=context,
        rew=0,
        statTitles=statTitles,
        statValues=statValues,
    )

    # Tim Agent dang yeu cau quyet dinh
    agent = env._agent_selection_fn(alpyne_obs)

    # Lay Behavior cua Agent
    behavior = env.agent_behaviors[agent]

    # Cap nhat context (mang luoi, vi tri...)
    if behavior.is_for_learning(alpyne_obs):
        raise RuntimeError(
            f"Agent {agent} requires an RL policy"
        )

    # Tinh Action theo thuat toan cua Tobias
    return behavior.get_action(alpyne_obs, time=time)
