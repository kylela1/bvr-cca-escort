"""
This is the b_ace_py package, which provides the Python interface to the B-ACE (Blue Air
Combat Environment) Godot simulation. The package contains the low-level socket-based
communication layer (godot_env.py), the PettingZoo-compatible multi-agent wrapper
(B_ACE_GodotPettingZooWrapper.py), utility functions for config loading and action space
processing (utils.py), enriched observation computation with ablation support
(enriched_observation_wrapper_ablation.py), and pursuit-evasion geometry calculations
(pursuit_evasion_features.py).

Within the broader pipeline, the rl_files training and evaluation scripts import from this
package to instantiate and interact with the Godot environment. The wrapper stack in
train_bace_clean.py builds on B_ACE_GodotPettingZooWrapper and optionally applies
EnrichedObservationWrapper and SplitNormalizationWrapper from this package.

This package has no significant hyperparameters of its own. Configuration is handled by the
JSON scenario config files (SimpleExample_B_ACE_config.json and Default_B_ACE_config.json)
which are loaded by utils.load_b_ace_config and control the Godot simulation parameters
such as aircraft performance, weapon properties, and scenario setup.
"""
