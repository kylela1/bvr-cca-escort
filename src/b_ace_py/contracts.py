"""
Declares, as plain Python data, the Godot <-> Python interface contracts that were previously
implicit: (1) the raw observation vector's label order, which several files (most notably
b_ace_py/enriched_observation_wrapper_ablation.py and rl_files/aggressive_hunter_fixed.py)
assume by hardcoded numeric index rather than by name, and (2) the info-dict keys Godot's
SimManager.gd actually emits, which reward wrappers and evaluate_experiment_seeds.py read by
string key with no central declaration of what exists.

Before this module existed, nothing checked that Godot's *actual* reported observation labels
(already printed to the console at every environment construction -- see
B_ACE_GodotPettingZooWrapper.py) matched what the feature-computation code assumes. If the
Godot-side observation order ever changed (plausible during 2v2 or missile-evasion follow-on
work), every file relying on positional indices would silently start reading the wrong field
into the wrong computation -- no error, just quietly wrong numbers. This module turns that into
a loud failure at environment-construction time instead (see assert_observation_labels()).

EXPECTED_OBSERVATION_LABELS was transcribed verbatim from a real B_ACE_v0.1.exe run's console
output (agent key "101", the blue agent). See IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md
Task 6.

EXPECTED_INFO_KEYS was enumerated by reading Godot_Air_Combat/SimManager.gd's
_get_info_from_agents() (the single source of truth for what Godot actually emits) rather than
by grepping the Python side's info.get(...) call sites, since the Python side reading a key
that Godot never emits is exactly the kind of silent-drift bug this module exists to catch. Not
every key is present on every step -- several (termination_reason, mission_outcome,
mission_success, all_red_destroyed, all_blue_destroyed) are only populated at episode end. This
module does not assert their presence for that reason; check_info_keys() only warns about keys
that are NOT in this known set, to surface genuinely new/renamed Godot-side fields.
"""

from typing import Iterable

EXPECTED_OBSERVATION_LABELS = [
    "own_x_pos",
    "own_z_pos",
    "own_altitude",
    "own_dist_target",
    "own_aspect_angle_target",
    "own_current_hdg",
    "own_current_speed",
    "own_missiles",
    "own_in_flight_missile",
    "hvaa_dist",
    "hvaa_alt_diff",
    "hvaa_angle_off",
    "hvaa_heading",
    "hvaa_detected",
    "track_alt_diff_201",
    "track_aspect_angle_201",
    "track_angle_off_201",
    "track_dist_201",
    "track_dist2go_201",
    "track_own_missile_RMax_201",
    "track_own_missile_Nez_201",
    "track_enemy_missile_RMax_201",
    "track_enemy_missile_Nez_201",
    "track_threat_factor_201",
    "track_offensive_factor_201",
    "track_is_missile_support_201",
    "track_detected_201",
]

##############################
# Keys Godot's SimManager.gd's _get_info_from_agents() emits, per-step or at episode end, plus
# keys B_ACE_GodotPettingZooWrapper.py itself synthesizes from those before Python code ever
# sees the info dict.
#
# NOTE (found via this contract's own check_info_keys() the first time it ran against a real
# environment): "missile_fired" is synthesized in B_ACE_GodotPettingZooWrapper.py by comparing
# `info[agent]["missiles_fired_total"]` against its previous value -- but Godot's SimManager.gd
# never actually emits a "missiles_fired_total" key anywhere, so that read always falls back to
# its default of 0 and "missile_fired" is always False, unconditionally. Currently harmless:
# nothing in rl_files/ or b_ace_py/ reads "missile_fired" downstream (confirmed via grep), so
# this is a real but presently-inert bug. Worth fixing before anything (e.g. missile-evasion
# work) comes to depend on it -- left as-is here since fixing it is a behavior change outside
# this task's scope (see IMPROVEMENT_PLAN_2_WRAPPERS_AND_DEPENDENCIES.md's ground rules).
##############################
EXPECTED_INFO_KEYS = frozenset([
    # Every step
    "missile_fired",  # see NOTE above -- currently always False
    "episode_over",
    "episode_steps",
    "sim_time_sec",
    "hvaa_alive",
    "hvaa_destroyed",
    "blue_killed_total",
    "red_killed_total",
    "blue_kills_this_step",
    "red_kills_this_step",
    "blue_killed_event",
    "red_killed_event",
    "blue_killed",
    "red_killed",
    "blue_missiles_fired",
    "red_missiles_fired",
    "blue_pk",
    "red_pk",
    "min_separation_nm",
    "blue_shot_first",
    "red_shot_first",
    "no_shots_fired",
    # Episode-end only
    "all_red_destroyed",
    "all_blue_destroyed",
    "termination_reason",
    "mission_outcome",
    "mission_success",
])


def assert_observation_labels(actual_labels: Iterable[str], agent_key: str = "101") -> None:
    """
    Raise a clear error if Godot's actually-reported observation labels (for the given agent
    key) don't match EXPECTED_OBSERVATION_LABELS exactly, in order. Positional index code
    (enriched_observation_wrapper_ablation.py's IDX_* constants, aggressive_hunter_fixed.py's
    get_default_obs_indices()) depends on this order being stable.
    """
    actual = list(actual_labels)
    expected = EXPECTED_OBSERVATION_LABELS

    if actual == expected:
        return

    if len(actual) != len(expected):
        raise AssertionError(
            f"[contracts] Observation label count mismatch for agent '{agent_key}': "
            f"expected {len(expected)} labels, Godot reported {len(actual)}.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"This means b_ace_py/contracts.py's EXPECTED_OBSERVATION_LABELS is out of date "
            f"with the current Godot build, or the Godot build itself changed its observation "
            f"layout -- either way, every file that reads observation indices by position "
            f"(enriched_observation_wrapper_ablation.py, aggressive_hunter_fixed.py) would "
            f"silently compute wrong features if this check were skipped."
        )

    diffs = [
        f"  index {i}: expected {e!r}, got {a!r}"
        for i, (e, a) in enumerate(zip(expected, actual))
        if e != a
    ]
    raise AssertionError(
        f"[contracts] Observation label mismatch for agent '{agent_key}' at "
        f"{len(diffs)} index(es):\n" + "\n".join(diffs) + "\n"
        f"This means b_ace_py/contracts.py's EXPECTED_OBSERVATION_LABELS is out of date "
        f"with the current Godot build, or the Godot build itself changed its observation "
        f"layout -- either way, every file that reads observation indices by position "
        f"(enriched_observation_wrapper_ablation.py, aggressive_hunter_fixed.py) would "
        f"silently compute wrong features if this check were skipped."
    )


def check_info_keys(info: dict, source: str = "info") -> None:
    """
    Warn (do not raise) about any key in `info` that isn't in EXPECTED_INFO_KEYS. Unlike
    observation labels, the info dict legitimately varies key-by-key across steps (episode-end
    keys aren't present mid-episode), so an unknown key is a signal worth surfacing but not
    necessarily a hard failure the way an observation label mismatch is.
    """
    unexpected = set(info.keys()) - EXPECTED_INFO_KEYS
    if unexpected:
        print(
            f"[contracts] WARNING: {source} contains key(s) not in EXPECTED_INFO_KEYS: "
            f"{sorted(unexpected)}. If Godot's SimManager.gd started emitting a new field, "
            f"add it to b_ace_py/contracts.py's EXPECTED_INFO_KEYS."
        )
