"""
This file implements a rule-based expert agent that uses pure-pursuit intercept logic to
engage and destroy the red (enemy) aircraft. The expert always turns directly toward the
enemy, flies straight at them, and fires whenever within range and roughly aligned. It
intentionally has no defensive maneuvers, wide turns, or survivability logic -- its sole
objective is to kill the red aircraft as quickly as possible.

Within the training pipeline, this expert is used by both ExpertAlignmentWrapper and
ExpertActionBlendingWrapper to provide demonstration-quality actions for reward shaping or
action blending. The get_default_obs_indices function maps observation vector indices to
the named fields the expert needs (e.g., heading error, range, altitude), enabling the
expert to interpret the raw observation array produced by the Godot environment.

Key configurable parameters (AggressiveHunterFixed.__init__): fire_range_min/fire_range_max
set the normalized range envelope within which the expert will fire (default: 0.08-0.40).
fire_cooldown_steps sets the minimum steps between consecutive fire commands (default: 100).
fire_cone sets the maximum aspect-angle-off magnitude at which the expert will fire (default:
0.55, on a normalized -1..1 scale). pursuit_gain and max_turn_rate control how aggressively
the expert turns toward the enemy (defaults: 2.0 and 0.6). pursuit_memory_steps (default: 400,
set directly in __init__, not a constructor parameter) is how long the expert continues
pursuing a last-known enemy position after losing track before reverting to patrol.
"""

import numpy as np
from typing import Dict, Any, Optional


class AggressiveHunterFixed:
    """
    TRUE KILL-PRIORITY aggressive hunter.

    Key fix: REMOVED all "enemy behind" special handling.
    Now it just always turns toward the enemy and flies at them.
    No more orbiting because there's no mode-switching.
    """

    def __init__(self,
                 fire_range_min: float = 0.08,
                 fire_range_max: float = 0.40,
                 fire_cooldown_steps: int = 100,
                 fire_cone: float = 0.55,

                 pursuit_gain: float = 2.0,
                 max_turn_rate: float = 0.6,
                 angle_deadband: float = 0.02,

                 debug: bool = False):

        self.fire_range_min = fire_range_min
        self.fire_range_max = fire_range_max
        self.fire_cooldown_steps = fire_cooldown_steps
        self.fire_cone = fire_cone
        self.pursuit_gain = pursuit_gain
        self.max_turn_rate = max_turn_rate
        self.angle_deadband = angle_deadband
        self.debug = debug

        self.step_count = 0
        self.fire_cooldown = 0

        self.last_detection_step = -1000
        self.pursuit_memory_steps = 400
        self.last_known_angle_off = 0.0

        self.fire_count = 0
        self.engage_count = 0

    def reset(self):
        self.step_count = 0
        self.fire_cooldown = 0
        self.last_detection_step = -1000
        self.last_known_angle_off = 0.0

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_steps': self.step_count,
            'fire_count': self.fire_count,
            'engage_count': self.engage_count,
        }

    def get_action(self, obs: np.ndarray, obs_indices: Dict[str, int]) -> np.ndarray:
        self.step_count += 1
        ##############################
        # Decrement fire cooldown if it is active
        ##############################
        if self.fire_cooldown > 0:
            self.fire_cooldown -= 1

        n = len(obs)

        def get_val(name: str, default: float = 0.0) -> float:
            idx = obs_indices.get(name)
            ##############################
            # Check if index is valid before accessing observation
            ##############################
            if idx is not None and 0 <= idx < n:
                return float(obs[idx])
            return default

        enemy_detected = get_val('enemy_detected', 0.0) > 0.5
        distance = get_val('distance_to_enemy', -1.0)
        angle_off = get_val('angle_off_to_enemy', 0.0)
        missiles = get_val('own_missiles', 6.0)
        missile_in_flight = get_val('own_in_flight_missile', 0.0) > 0.5
        own_hdg = get_val('own_hdg', 0.0)

        valid_enemy = enemy_detected and distance >= 0

        ##############################
        # Update pursuit memory if enemy is currently detected
        ##############################
        if valid_enemy:
            self.last_detection_step = self.step_count
            self.last_known_angle_off = angle_off

        steps_since_detection = self.step_count - self.last_detection_step
        in_pursuit = steps_since_detection < self.pursuit_memory_steps

        ##############################
        # If enemy is detected, enter engage mode with pure pursuit
        ##############################
        if valid_enemy:
            self.engage_count += 1

            ##############################
            # If nearly aligned with enemy, fly straight; otherwise turn toward enemy
            ##############################
            if abs(angle_off) < self.angle_deadband:
                turn = 0.0
            else:
                turn = np.clip(-angle_off * self.pursuit_gain,
                              -self.max_turn_rate, self.max_turn_rate)

            g_force = 0.8
            altitude = 0.1

            ##############################
            # Check if conditions are met to fire a missile
            ##############################
            fire = 0.0
            if missiles > 0 and not missile_in_flight and self.fire_cooldown <= 0:
                in_range = self.fire_range_min < distance < self.fire_range_max
                in_cone = abs(angle_off) < self.fire_cone

                ##############################
                # Fire if both in range and within firing cone
                ##############################
                if in_range and in_cone:
                    fire = 1.0
                    self.fire_cooldown = self.fire_cooldown_steps
                    self.fire_count += 1

                    ##############################
                    # Print debug information when firing
                    ##############################
                    if self.debug:
                        print(f"[FIRE!] d={distance:.2f} ang={angle_off:.2f}")

                ##############################
                # Print debug information about why we didn't fire
                ##############################
                elif self.debug and self.step_count % 100 == 0:
                    print(f"[NO FIRE] in_range={in_range}(d={distance:.2f}) in_cone={in_cone}(ang={angle_off:.2f})")

        ##############################
        # If enemy lost but recently detected, continue toward last known position
        ##############################
        elif in_pursuit:
            turn = np.clip(-self.last_known_angle_off * 1.5,
                          -self.max_turn_rate, self.max_turn_rate)
            g_force = 0.75
            altitude = 0.0
            fire = 0.0

            ##############################
            # Print debug information about pursuit mode
            ##############################
            if self.debug and self.step_count % 200 == 0:
                print(f"[PURSUIT] last_ang={self.last_known_angle_off:.2f}")

        ##############################
        # If no contact and pursuit expired, search for enemy by flying north
        ##############################
        else:
            heading_error = own_hdg

            ##############################
            # If not aligned with north, turn toward it
            ##############################
            if abs(heading_error) > 0.03:
                turn = np.clip(-heading_error * 2.0, -self.max_turn_rate, self.max_turn_rate)
                g_force = 0.6
            else:
                turn = 0.0
                g_force = 0.55

            altitude = 0.0
            fire = 0.0

            ##############################
            # Print debug information about hunt mode
            ##############################
            if self.debug and self.step_count % 300 == 0:
                print(f"[HUNT] hdg={own_hdg:.2f}")

        return np.array([turn, altitude, g_force, fire], dtype=np.float32)


def get_default_obs_indices() -> Dict[str, int]:
    """Default observation indices matching Godot B-ACE output (27 dims)."""
    return {
        'own_x': 0,
        'own_z': 1,
        'own_altitude': 2,
        'dist_to_target': 3,
        'aspect_to_target': 4,
        'own_hdg': 5,
        'own_speed': 6,
        'own_missiles': 7,
        'own_in_flight_missile': 8,
        'distance_to_hvaa': 9,
        'hvaa_alt_diff': 10,
        'hvaa_angle_off': 11,
        'hvaa_heading': 12,
        'hvaa_detected': 13,
        'altitude_diff_enemy': 14,
        'aspect_angle_to_enemy': 15,
        'angle_off_to_enemy': 16,
        'distance_to_enemy': 17,
        'dist2go_enemy': 18,
        'own_missile_rmax': 19,
        'own_missile_nez': 20,
        'enemy_missile_rmax': 21,
        'enemy_missile_nez': 22,
        'defensive_factor': 23,
        'offensive_factor': 24,
        'is_missile_support': 25,
        'enemy_detected': 26,
    }


if __name__ == "__main__":
    print("="*60)
    print("AGGRESSIVE HUNTER - TRUE KILL PRIORITY (NO ORBIT)")
    print("="*60)
    print("""
    PROBLEM FIXED: Orbiting/spinning after firing

    ROOT CAUSE: Previous version had "enemy behind" logic that triggered
    wide sweeping turns at reduced turn rate. This caused orbiting because:
    1. Fire when enemy is ahead
    2. Geometry shifts, enemy crosses 0.6 angle threshold
    3. Enters "wide turn" mode -> starts orbiting

    FIX: REMOVED all "enemy behind" special handling.

    New behavior is PURE PURSUIT:
    - Always turn directly toward enemy
    - Same turn rate regardless of enemy position
    - No mode switching, no defensive maneuvers
    - Just point at them and fly

    Result: Identify -> Fire -> Fly STRAIGHT at red -> Fire again

    Key parameters:
    - pursuit_gain: 2.0 (how aggressively to turn toward enemy)
    - max_turn_rate: 0.6 (single rate, no slow/fast modes)
    - fire_range: 0.08-0.40
    - fire_cone: ~100 degrees
    - cooldown: 100 steps
    - g_force: 0.8 in engage (high speed pursuit)

    Experts:
      AggressiveHunterFixed   - Main version, pure pursuit (the only one imported/used
                                 elsewhere in this repo)

    Test:
      python test_expert_standalone.py --expert hunter_fixed --episodes 10 --render --verbose
    """)
