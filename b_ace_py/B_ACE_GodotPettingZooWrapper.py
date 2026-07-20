"""
This file implements B_ACE_GodotPettingZooWrapper, a PettingZoo ParallelEnv that wraps the
low-level GodotEnv socket interface to provide a multi-agent gymnasium-compatible
environment for the B-ACE air combat simulation. It handles agent registration, observation
and action space construction from the Godot environment's reported configuration, and
translates between PettingZoo's parallel API (dict-based obs/actions keyed by agent name)
and the underlying Godot socket protocol.

Within the broader pipeline, this wrapper is instantiated by SingleAgentBACEEnv in
train_bace_clean.py (which extracts a single agent's perspective for SB3 compatibility) and
directly by evaluate_experiment_seeds.py for evaluation. It inherits from both GodotEnv
(for Godot communication) and PettingZoo's ParallelEnv (for multi-agent API compliance).
The JSON scenario config file is passed through to the Godot engine to configure the
simulation scenario.

Key configurable parameters: env_path specifies the Godot executable location, seed sets
the simulation random seed (passed through to Godot), convert_action_space toggles between
raw and processed action spaces, and additional keyword arguments are forwarded to the
Godot environment configuration.
"""

from .godot_env import GodotEnv
from pettingzoo.utils import ParallelEnv

from .utils import ActionSpaceProcessor, convert_macos_path
import numpy as np
import atexit
from sys import platform
from gymnasium import spaces
import random


class B_ACE_GodotPettingZooWrapper(GodotEnv, ParallelEnv):
    metadata = {'render.modes': [], 'name': "godot_rl_multi_agent"}

    def __init__(self,
                env_path: str = None,
                seed: int = 0,
                convert_action_space: bool = False,
                device: str = "cpu",
                **config_kwargs):

        self.device = device

        self.env_config = config_kwargs.get("EnvConfig", "")
        self.env_path       = self.env_config.get("env_path", "./bin/BVR.exe")
        self.show_window    = int(self.env_config.get("renderize", 1))
        self._seed          = int(self.env_config.get("seed", 1))
        self.action_repeat  = int(self.env_config.get("action_repeat", 20))
        self.action_type    = self.env_config.get("action_type", "Low_Level_Continuous")
        self.speedup        = int(self.env_config.get("speed_up", 1000))
        self.parallel_envs  = int(self.env_config.get("parallel_envs", 1))

        self.agents_config = config_kwargs.get("AgentsConfig", "")
        self._num_agents = int(self.agents_config["blue_agents"].get("num_agents", 1))

        self.share_states  = int(self.agents_config["blue_agents"].get("share_states", 1))
        self.share_tracks =  int(self.agents_config["blue_agents"].get("share_tracks", 1))

        self.additional_config = self.env_config.get("additional_config", "")

        self.port = int(self.env_config.get("port", 11008))

        self.proc = None

        ##############################
        # Check if a valid environment path is provided and is not in debug mode
        ##############################
        if self.env_path is not None and self.env_path != "debug":
            self.env_path = self._set_platform_suffix(self.env_path)

            self.check_platform(self.env_path)

            self._launch_env(self.env_path, self.port, self.show_window == 1, None, self._seed, self.action_repeat, self.speedup)
        else:
            print("No game binary has been provided, please press PLAY in the Godot editor")

        self.host_binding = config_kwargs.get("host_binding", False)

        self.connection = self._start_server()
        self.num_envs = None

        self._handshake()

        self.action_spaces = []
        self.observation_spaces = []

        self.send_sim_config(self.env_config, self.agents_config)

        self.action_spaces = []
        self.observation_spaces = []

        env_info = self._get_env_info()

        self.observation_labels = env_info["observation_labels"]
        first_key = str(101)
        self.observation_labels_flat = list(self.observation_labels[first_key])
        ##############################
        # Check if observation labels have been printed before
        ##############################
        if not hasattr(self, "_printed_observation_labels"):
            print("\n=== B-ACE Observation Labels (raw, from Godot) ===")
            ##############################
            # Iterate through all observation labels for each agent and print them
            ##############################
            for agent_id, labels in self.observation_labels.items():
                print(f"\nAgent key: {agent_id}")
                ##############################
                # Print each label with its index number
                ##############################
                for i, label in enumerate(labels):
                    print(f"{i:>2}: {label}")
            print("=== End Observation Labels ===\n")
            self._printed_observation_labels = True
        self.action_spaces = env_info["action_spaces"]

        self.tuple_action_spaces = [
            spaces.Tuple([v for _, v in action_space.items()]) for action_space in self.action_spaces
        ]
        self.action_space_processor = ActionSpaceProcessor(self.tuple_action_spaces[0], convert_action_space)

        self.agent_policy_names

        atexit.register(self._close)

        ##############################
        # Initialize agents list for PettingZoo Parallel environment
        ##############################
        self.agents = [f'agent_{i}' for i in range(self._num_agents)]
        self.possible_agents = self.agents[:]

        self._prev_missiles_fired = {a: 0 for a in self.possible_agents}

        ##############################
        # Create observation mapping for each agent to label indices
        ##############################
        self.obs_map= {agent : {label: index for index, label in enumerate(self.observation_labels[str(101 + i)])}  for i,agent in enumerate(self.possible_agents)}               


    def get_hvaa_indices(self, agent_name: str):

        labels = None
        ##############################
        # Check if observation labels attribute exists
        ##############################
        if hasattr(self, "observation_labels"):
            labels = self.observation_labels.get("101", None)

        ##############################
        # Return None if labels not found for agent 101
        ##############################
        if labels is None:
            print("[get_hvaa_indices] WARNING: no labels for '101'; returning None")
            return None

        label_to_idx = {name: idx for idx, name in enumerate(labels)}

        required = ["hvaa_dist", "hvaa_alt_diff", "hvaa_angle_off", "hvaa_detected"]
        missing = [r for r in required if r not in label_to_idx]
        ##############################
        # Check if any required HVAA labels are missing
        ##############################
        if missing:
            print("[get_hvaa_indices] WARNING: missing HVAA labels:", missing)
            return None

        return {
            "hvaa_dist": label_to_idx["hvaa_dist"],
            "hvaa_alt_diff": label_to_idx["hvaa_alt_diff"],
            "hvaa_angle_off": label_to_idx["hvaa_angle_off"],
            "hvaa_detected": label_to_idx["hvaa_detected"],
        }


        self.agent_idx = [ {agent : i} for i, agent in enumerate(self.possible_agents)]

        self.observation_space = self.observation_spaces[0]["obs"]
        self.action_space = self.action_spaces[0]["input"]

        self._cumulative_rewards = {agent : 0  for agent in self.possible_agents}
        self.rewards =  {agent : 0  for agent in self.possible_agents}
        self.terminations =  {agent : False  for agent in self.possible_agents}
        self.truncations =  {agent : False  for agent in self.possible_agents}
        self.observations =  {agent : []  for agent in self.possible_agents}
        self.info =  {agent : []  for agent in self.possible_agents}
        self.infos =  {agent : []  for agent in self.possible_agents}

        self._ep_step = 0
        self._printed_ep_end = False
                            
    def send_sim_config(self, _env_config, _agents_config):
        message = {"type": "config"}
        message["agents_config"] = _agents_config
        message["env_config"] = _env_config

        self._send_as_json(message)

        response = self._get_dict_json_message()

        ##############################
        # Check if response type is acknowledgment (ack)
        ##############################
        if response.get("type") != "ack":
            print(f"WARNING: Expected 'ack' response, got '{response.get('type')}'")

    
    def reset(self, seed=0, options=None):
        obs_dict, _info_from_base = super().reset()

        self.observations = {}
        self.info = {}

        ##############################
        # Process observations for each possible agent
        ##############################
        for agent_name in self.possible_agents:
            ##############################
            # Check if agent observations exist in response from Godot
            ##############################
            if agent_name not in obs_dict:
                print(f"WARNING: {agent_name} not in obs_dict from Godot; filling zeros")
                self.observations[agent_name] = {
                    "obs": [],
                    "mask": [True for _ in range(4)]
                }
                self.info[agent_name] = {}
                continue

            raw_obs = obs_dict[agent_name]

            self.observations[agent_name] = {
                "obs": raw_obs,
                "mask": [True for _ in range(4)]
            }

            self.info[agent_name] = {}

        self._ep_step = 0
        self._printed_ep_end = False

        return self.observations, self.info
    

    
    
    def _observation_space(self, agent):        
        return self.observation_spaces[agent]
    
    def action_space(self, agent = None):        
        return self.action_space_processor.action_space
    
    def seed(self, _seed):
        self.seed = _seed

    def step(self, actions, order_ij=True):
        ##############################
        # Check action type and format actions for continuous control
        ##############################
        if self.action_type == "Low_Level_Continuous":
            godot_actions = []
            ##############################
            # Build action list with continuous values for each agent
            ##############################
            for agent_name in self.possible_agents:
                a = np.asarray(actions[agent_name], dtype=np.float32)
                godot_actions.append([a])

        ##############################
        # Check if action type is discrete and format actions accordingly
        ##############################
        elif self.action_type == "Low_Level_Discrete":
            ordered = []
            ##############################
            # Decode discrete actions for each agent
            ##############################
            for agent_name in self.possible_agents:
                ordered.append(self.decode_action(actions[agent_name]))
            godot_actions = [[np.asarray(ordered[i], dtype=np.int32)]
                            for i in range(len(ordered))]
        else:
            print("GodotPZWrapper::Error:: Unknown Actions Type -> ", self.action_type)

        obs, reward, dones, truncs, info = super().step(godot_actions, order_ij=order_ij)

        self._ep_step += 1

        ##############################
        # Convert info to proper dict format if it's a list or not a dict
        ##############################
        if isinstance(info, list):
            info_dict = {}
            ##############################
            # Map info list items to agent names
            ##############################
            for i, agent_name in enumerate(self.agents):
                ##############################
                # Check if info item exists and is a dict
                ##############################
                if i < len(info) and isinstance(info[i], dict):
                    info_dict[agent_name] = info[i]
                else:
                    info_dict[agent_name] = {}
            info = info_dict
        elif not isinstance(info, dict):
            info = {agent_name: {} for agent_name in self.agents}

        ##############################
        # Check if episode_over flag is set in any agent's info
        ##############################
        episode_over_from_info = any(
            bool(info.get(a, {}).get("episode_over", False))
            for a in self.possible_agents
        )

        ##############################
        # Check if any agent is done or truncated from Godot's signals
        ##############################
        _any_done = any(bool(dones.get(a, False)) for a in self.possible_agents)
        _any_trunc = any(bool(truncs.get(a, False)) for a in self.possible_agents)

        ##############################
        # Episode ends if any termination condition is true
        ##############################
        _episode_end = episode_over_from_info or _any_done or _any_trunc

        ##############################
        # Set final termination flags for all agents based on combined condition
        ##############################
        for a in self.possible_agents:
            dones[a] = _episode_end
            truncs[a] = False

        ##############################
        # Derive per-step missile_fired flag for each agent
        ##############################
        for agent_name in self.possible_agents:
            ai = info.get(agent_name, {})

            cur_total = ai.get("missiles_fired_total", 0)

            prev_total = self._prev_missiles_fired.get(agent_name, 0)
            fired_this_step = cur_total > prev_total

            ai["missile_fired"] = bool(fired_this_step)
            info[agent_name] = ai

            self._prev_missiles_fired[agent_name] = cur_total

        ##############################
        # On first episode end, print episode summary information
        ##############################
        if _episode_end and not self._printed_ep_end:
            self._printed_ep_end = True
            sim_time = None
            ##############################
            # Collect simulation time from agent info if available
            ##############################
            for a in self.possible_agents:
                ##############################
                # Check if agent info exists and contains sim_time
                ##############################
                if a in info and isinstance(info[a], dict):
                    sim_time = info[a].get("sim_time_sec") or info[a].get("sim_time") or sim_time

            ##############################
            # Collect termination reasons from all agents
            ##############################
            reasons = []
            for a in self.possible_agents:
                ai = info.get(a, {})
                r = ai.get("termination_reason") or ai.get("reason") or ai.get("end_reason")
                ##############################
                # Add reason to list if it exists
                ##############################
                if r:
                    reasons.append(f"{a}:{r}")
            reason_str = "; ".join(reasons) if reasons else "unknown"

            print(f"[EP-END] steps={self._ep_step} any_done={_any_done} any_trunc={_any_trunc} "
                f"sim_time={sim_time} reasons=[{reason_str}]")

        ##############################
        # Process observations into proper format for each agent
        ##############################
        self.observations = {agent_name : {"obs": _obs["obs"], "mask": [True for _ in range(4)]} for agent_name, _obs in obs.items()}

        ##############################
        # Aggregate termination and truncation flags for the environment
        ##############################
        self.terminations = _episode_end
        self.truncations = False
        self.rewards = 0.0

        ##############################
        # Sum rewards from all agents
        ##############################
        for i, agent in enumerate(self.possible_agents):
            self.rewards += reward[agent]

        ##############################
        # Print debug message when episode ends
        ##############################
        if _episode_end:
            print(f"[TERM-CHECK] _episode_end={_episode_end} → self.terminations={self.terminations}, self.truncations={self.truncations}")

        self.info = info
        return self.observations, self.rewards, self.terminations, self.truncations, self.info


    def _process_obs(self, response_obs):
        return response_obs
    
    def last(self, env=None):

        """Return the last observations, rewards, and done status."""
        return (
            self.observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.info,
        )

    def decode_action(self, encoded_action):
        ##############################
        # Decode discrete action into continuous components
        ##############################
        turn_input = encoded_action % 5
        level_input = (encoded_action // 5.0) % 5
        fire_input = (encoded_action // 25.0) % 2
        return np.array([fire_input, level_input, turn_input])