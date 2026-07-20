"""
This file provides shared utility functions used across the b_ace_py package and the
rl_files training scripts. It includes load_b_ace_config for reading and parsing B-ACE
JSON scenario configuration files, ActionSpaceProcessor for converting between discrete and
continuous action space representations, convert_macos_path for resolving macOS .app bundle
paths to their internal binaries, and helper functions lod_to_dol / dol_to_lod for
converting between list-of-dicts and dict-of-lists formats.

Within the broader pipeline, load_b_ace_config is called by train_bace_clean.py and
evaluate_experiment_seeds.py to parse the JSON scenario file that defines the simulation
parameters. ActionSpaceProcessor is used by B_ACE_GodotPettingZooWrapper to translate
between the Godot simulation's action format and the gymnasium action space. convert_macos_path
is used by godot_env.py to locate the Godot binary inside a macOS application bundle.

This file has no significant hyperparameters. The behavior of load_b_ace_config is
determined entirely by the contents of the JSON config file it reads. ActionSpaceProcessor
adapts to whatever action space definition the Godot environment provides at runtime.
"""

import importlib
import re

import gymnasium as gym
import numpy as np
import json


def lod_to_dol(lod):
    return {k: [dic[k] for dic in lod] for k in lod[0]}


def dol_to_lod(dol):
    return [dict(zip(dol, t)) for t in zip(*dol.values())]


def convert_macos_path(env_path):
    """
    On MacOs the user is supposed to provide a application.app file to env_path.
    However the actual binary is in application.app/Contents/Macos/application.
    This helper function converts the path to the path of the actual binary.

    Example input: ./Demo.app
    Example output: ./Demo.app/Contents/Macos/Demo
    """

    filenames = re.findall(r"[^\/]+(?=\.)", env_path)
    assert len(filenames) == 1, "An error occured while converting the env path for MacOS."
    return env_path + "/Contents/MacOS/" + filenames[0]


class ActionSpaceProcessor:
    """Convert tuple action distributions to a single continuous or multi-discrete action distribution."""
    def __init__(self, action_space: gym.spaces.Tuple, convert) -> None:
        self._original_action_space = action_space
        self._convert = convert
        self._all_actions_discrete: bool = all(isinstance(space, gym.spaces.Discrete) for space in action_space.spaces)
        self._only_one_action_space: bool = len(action_space) == 1

        ##############################
        # Check if we have only one discrete action space, which requires no conversion
        ##############################
        if self._only_one_action_space and self._all_actions_discrete:
            self.converted_action_space = action_space[0]
            return

        space_size = 0

        ##############################
        # If conversion is enabled, process the action space to create a unified representation
        ##############################
        if convert:
            use_multi_discrete_spaces = False
            multi_discrete_spaces = np.array([])
            ##############################
            # Check if the action space is a tuple of spaces
            ##############################
            if isinstance(action_space, gym.spaces.Tuple):
                ##############################
                # If all spaces are discrete, prepare them for multi-discrete conversion
                ##############################
                if self._all_actions_discrete:
                    use_multi_discrete_spaces = True
                    ##############################
                    # Collect the number of discrete actions for each space
                    ##############################
                    for space in action_space.spaces:
                        multi_discrete_spaces = np.append(multi_discrete_spaces, space.n)
                ##############################
                # Otherwise, handle mixed continuous and discrete spaces
                ##############################
                else:
                    ##############################
                    # Process each space in the tuple to calculate total continuous size
                    ##############################
                    for space in action_space.spaces:
                        ##############################
                        # If the space is continuous (Box), add its dimensionality to total size
                        ##############################
                        if isinstance(space, gym.spaces.Box):
                            assert len(space.shape) == 1
                            space_size += space.shape[0]
                        ##############################
                        # If the space is discrete, validate it and add 1 to size (only binary supported with continuous)
                        ##############################
                        elif isinstance(space, gym.spaces.Discrete):
                            ##############################
                            # Reject discrete actions with more than 2 options when mixed with continuous
                            ##############################
                            if space.n > 2:
                                raise NotImplementedError(
                                    "Discrete actions with size larger than 2 "
                                    "are currently not supported if used together with continuous actions."
                                )
                            space_size += 1
                        ##############################
                        # Reject any other space types
                        ##############################
                        else:
                            raise NotImplementedError
            ##############################
            # Reject Dict action spaces as not yet supported
            ##############################
            elif isinstance(action_space, gym.spaces.Dict):
                raise NotImplementedError
            ##############################
            # Handle single action spaces (not tuple)
            ##############################
            else:
                assert isinstance(action_space, (gym.spaces.Box, gym.spaces.Discrete))
                return

            ##############################
            # Create the appropriate converted action space based on the collected information
            ##############################
            if use_multi_discrete_spaces:
                self.converted_action_space = gym.spaces.MultiDiscrete(multi_discrete_spaces)
            ##############################
            # Otherwise create a continuous Box action space
            ##############################
            else:
                self.converted_action_space = gym.spaces.Box(-1, 1, shape=[space_size])

    @property
    def action_space(self):
        ##############################
        # Return original action space if no conversion was requested
        ##############################
        if not self._convert:
            return self._original_action_space

        return self.converted_action_space

    def to_original_dist(self, action):
        ##############################
        # If conversion is disabled, return the action as-is
        ##############################
        if not self._convert:
            return action
        ##############################
        # If there is only one discrete action space, wrap the action in a list
        ##############################
        elif self._only_one_action_space and self._all_actions_discrete:
            return [action]

        original_action = []
        counter = 0

        integer_actions: bool = action.dtype == np.int64

        ##############################
        # Process each action space in the original tuple to reconstruct the original action format
        ##############################
        for space in self._original_action_space.spaces:
            ##############################
            # Handle continuous (Box) action spaces
            ##############################
            if isinstance(space, gym.spaces.Box):
                assert len(space.shape) == 1
                ##############################
                # Extract continuous actions for each agent in a multi-agent scenario
                ##############################
                for act in action:
                    original_action.append(act[counter : counter + space.shape[0]])
                counter += space.shape[0]

            ##############################
            # Handle discrete action spaces
            ##############################
            elif isinstance(space, gym.spaces.Discrete):
                discrete_actions = None

                ##############################
                # If actions are integers, extract the discrete action directly
                ##############################
                if integer_actions:
                    discrete_actions = action[:, counter]
                ##############################
                # If actions are floats, convert them to binary discrete actions
                ##############################
                else:
                    ##############################
                    # Validate that discrete actions with more than 2 options are not used
                    ##############################
                    if space.n > 2:
                        raise NotImplementedError(
                            "Discrete actions with size larger than "
                            "2 are currently not implemented for this algorithm."
                        )
                    ##############################
                    # Convert float values to binary (0 or 1) based on threshold of 0.0
                    ##############################
                    discrete_actions = np.greater(action[:, counter], 0.0)
                    discrete_actions = discrete_actions.astype(np.float32)

                original_action.append(discrete_actions)
                counter += 1

            ##############################
            # Reject any other space types
            ##############################
            else:
                raise NotImplementedError

        return original_action


def can_import(module_name):
    return not cant_import(module_name)


def cant_import(module_name):
    ##############################
    # Try to import the module and return False if successful, True if ImportError is raised
    ##############################
    try:
        importlib.import_module(module_name)
        return False
    ##############################
    # If the module cannot be imported, return True
    ##############################
    except ImportError:
        return True

def load_b_ace_config(config_json_file):

    ##############################
    # Attempt to read and parse the JSON configuration file
    ##############################
    try:
        with open(config_json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    ##############################
    # Handle the case where the configuration file does not exist
    ##############################
    except FileNotFoundError:
        print(f"Error: Config file '{config_json_file}' not found.")
    ##############################
    # Handle the case where the configuration file is not valid JSON
    ##############################
    except json.JSONDecodeError:
        print(f"Error: Config file '{config_json_file}' is not a valid JSON.")
    