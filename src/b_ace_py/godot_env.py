"""
This file implements the low-level GodotEnv class that manages socket-based communication
between Python and the Godot game engine. It handles launching the Godot executable as a
subprocess, establishing a TCP socket connection, and exchanging observations and actions as
JSON-encoded messages. The class manages the Godot process lifecycle including startup,
reset, step, and cleanup, and converts raw JSON observation data into numpy arrays with
properly defined gymnasium observation and action spaces.

Within the broader pipeline, GodotEnv is the base class for B_ACE_GodotPettingZooWrapper.
It is never instantiated directly by the training scripts -- instead,
B_ACE_GodotPettingZooWrapper inherits from it and adds multi-agent PettingZoo semantics on
top. The socket protocol sends action arrays to Godot each step and receives back
observation arrays, rewards, and done flags.

Key configurable parameters: the port for socket communication (auto-selected from
available ports), env_path specifies the path to the Godot executable, seed sets the
simulation random seed, and show_window controls whether the Godot rendering window is
displayed during training (typically disabled for headless training).
"""

import atexit
import json
import os
import pathlib
import socket
import subprocess
import time
from collections import OrderedDict
from sys import platform
from typing import Optional

import numpy as np
from gymnasium import spaces

from b_ace_py.utils import ActionSpaceProcessor, convert_macos_path


class GodotEnv:
    MAJOR_VERSION = "0"
    MINOR_VERSION = "1"
    DEFAULT_PORT = 11008
    DEFAULT_TIMEOUT = 60

    def __init__(
        self,
        env_path: str = None,
        port: int = DEFAULT_PORT,
        show_window: bool = False,
        seed: int = 0,
        framerate: Optional[int] = None,
        action_repeat: Optional[int] = None,
        speedup: Optional[int] = None,
        convert_action_space: bool = False,
        **kwargs,
    ):
        """
        Initialize a new instance of GodotEnv

        Args:
            env_path (str): path to the godot binary environment.
            port (int): Port number for communication.
            show_window (bool): flag to display Godot game window.
            seed (int): seed to initialize the environment.
            framerate (int): the framerate to run the Godot game at.
            action_repeat (int): the number of frames to repeat an action for.
            speedup (int): the factor to speedup game time by.
            convert_action_space (bool): flag to convert action space.
        """

        self.proc = None
        ##############################
        # Check if a valid environment path is provided (not None and not debug mode)
        ##############################
        if env_path is not None and env_path != "debug":
            env_path = self._set_platform_suffix(env_path)

            self.check_platform(env_path)
            self._launch_env(
                env_path,
                port,
                show_window,
                framerate,
                seed,
                action_repeat,
                speedup,
                **kwargs,
            )
        ##############################
        # If no environment path provided, notify user to start via Godot editor
        ##############################
        else:
            print("No game binary has been provided, please press PLAY in the Godot editor")

        self.port = port
        self.host_binding = kwargs.get("host_binding", False)
        self.connection = self._start_server()
        self.num_envs = None
        self._handshake()

        self.action_spaces = []
        self.observation_spaces = []

        self._get_env_info()

        self.observation_space = self.observation_spaces[0]

        self.tuple_action_spaces = [
            spaces.Tuple([v for _, v in action_space.items()]) for action_space in self.action_spaces
        ]
        self.action_space_processor = ActionSpaceProcessor(self.tuple_action_spaces[0], convert_action_space)

        self.agent_policy_names

        atexit.register(self._close)

    def _set_platform_suffix(self, env_path: str) -> str:
        """
        Set the platform suffix for the given environment path based on the platform.

        Args:
            env_path (str): The environment path.

        Returns:
            str: The environment path with the platform suffix.
        """
        suffixes = {
            "linux": ".x86_64",
            "linux2": ".x86_64",
            "darwin": ".app",
            "win32": ".exe",
        }
        suffix = suffixes[platform]
        return str(pathlib.Path(env_path).with_suffix(suffix))

    def check_platform(self, filename: str):
        """
        Check the platform and assert the file type

        Args:
            filename (str): Path of the file to check.

        Raises:
            AssertionError: If the file type does not match with the platform or file does not exist.
        """
        ##############################
        # Check if running on Linux platform and validate x86_64 file suffix
        ##############################
        if platform == "linux" or platform == "linux2":
            assert (
                pathlib.Path(filename).suffix == ".x86_64"
            ), f"Incorrect file suffix for {filename=} {pathlib.Path(filename).suffix=}. Please provide a .x86_64 file"
        ##############################
        # Check if running on macOS platform and validate .app file suffix
        ##############################
        elif platform == "darwin":
            assert (
                pathlib.Path(filename).suffix == ".app"
            ), f"Incorrect file suffix for {filename=} {pathlib.Path(filename).suffix=}. Please provide a .app file"
        ##############################
        # Check if running on Windows platform and validate .exe file suffix
        ##############################
        elif platform == "win32":
            assert (
                pathlib.Path(filename).suffix == ".exe"
            ), f"Incorrect file suffix for {filename=} {pathlib.Path(filename).suffix=}. Please provide a .exe file"
        ##############################
        # Raise error if platform is not recognized
        ##############################
        else:
            assert 0, f"unknown filetype {pathlib.Path(filename).suffix}"

        assert os.path.exists(filename)

    def from_numpy(self, action, order_ij=False):
        """
        Handles dict to tuple actions

        Args:
            action: The action to be converted.
            order_ij (bool): Order flag.

        Returns:
            list: The converted action.
        """
        result = []

        ##############################
        # Iterate through each agent/environment to convert and structure actions
        ##############################
        for agent_idx in range(self.num_envs):
            env_action = {}
            ##############################
            # Iterate through each action key for the current agent
            ##############################
            for j, k in enumerate(self.action_spaces[agent_idx].keys()):
                ##############################
                # Select action value based on order_ij flag (agent-first or action-first ordering)
                ##############################
                if order_ij is True:
                    v = action[agent_idx][j]
                ##############################
                # Else access action using action-first ordering
                ##############################
                else:
                    v = action[j][agent_idx]

                ##############################
                # Convert numpy arrays to lists; otherwise convert to int for JSON serialization
                ##############################
                if isinstance(v, np.ndarray):
                    env_action[k] = v.tolist()
                ##############################
                # Else convert value to int (numpy int32 cannot be JSON serialized)
                ##############################
                else:
                    env_action[k] = int(v)

            result.append(env_action)
        return result

    def step(self, action, order_ij=False):
        """
        Perform one step in the environment.

        Args:
            action: Action to be taken.
            order_ij (bool): Order flag.

        Returns:
            tuple: Tuple containing observation, reward, done flag, termination flag, and info.
        """
        self.step_send(action, order_ij=order_ij)
        return self.step_recv()

    def step_send(self, action, order_ij=False):
        """
        Send the action to the Godot environment.

        Args:
            action: Action to be sent.
            order_ij (bool): Order flag.
        """        
        action = self.action_space_processor.to_original_dist(action)

        message = {
            "type": "action",
            "action": self.from_numpy(action, order_ij=order_ij),
        }
        self._send_as_json(message)

    def step_recv(self):
        """
        Receive the step response from the Godot environment.

        Returns:
            tuple: Tuple containing observation, reward, done flag, termination flag, and info.
        """
        ##############################
        # Keep reading messages until we receive the observation payload
        ##############################
        while True:
            response = self._get_dict_json_message()

            rtype = response.get("type", None)
            ##############################
            # Break the loop once we receive a message containing observation data
            ##############################
            if "obs" in response:
                break

            print(f"[RESET DEBUG] skipping message type={rtype}, keys={list(response.keys())}")

        response["obs"] = self._process_obs(response["obs"])

        default_info = [{}] * len(response["done"])

        return (
            response["obs"],
            response["reward"],
            np.array(response["done"]).tolist(),
            np.array(response["done"]).tolist(),
            response.get("info", default_info),
        )

    def _process_obs(self, response_obs: dict):
        """
        Process observation data.

        Args:
            response_obs (dict): The response observation to be processed.

        Returns:
            dict: The processed observation data.
        """
        ##############################
        # Iterate through observation keys from first environment to identify 2D observations
        ##############################
        for k in response_obs[0].keys():
            ##############################
            # If observation key contains "2d", decode it from hex string for all environments
            ##############################
            if "2d" in k:
                ##############################
                # Decode 2D observation string for each environment/agent
                ##############################
                for sub in response_obs:
                    sub[k] = self._decode_2d_obs_from_string(sub[k], self.observation_space[k].shape)

        return response_obs

    def reset(self, seed=None):
        """
        Reset the Godot environment.

        Returns:
            dict: The initial observation data.
        """
        message = {
            "type": "reset",
        }
        self._send_as_json(message)
        max_msgs = 200
        obs_payload = None
        last = None

        ##############################
        # Loop through incoming messages with a maximum limit to find the reset observation
        ##############################
        for i in range(max_msgs):
            response = self._get_dict_json_message()
            last = response

            ##############################
            # Check if response contains observation data in standard format
            ##############################
            if isinstance(response, dict) and "obs" in response:
                obs_payload = response["obs"]
                break

            ##############################
            # Check if observation is nested inside a "returns" field (alternate Godot response format)
            ##############################
            if isinstance(response, dict) and isinstance(response.get("returns"), dict) and "obs" in response["returns"]:
                obs_payload = response["returns"]["obs"]
                break

            print(f"[RESET DEBUG] skipping msg {i+1}/{max_msgs}: type={response.get('type')} keys={list(response.keys())}")

        ##############################
        # If no observation payload received after max attempts, raise error
        ##############################
        if obs_payload is None:
            raise RuntimeError(
                f"Reset failed: never received 'obs' in {max_msgs} messages. "
                f"Last message type={last.get('type') if isinstance(last, dict) else type(last)} last={last}"
            )

        response["obs"] = self._process_obs(obs_payload)
        assert response["type"] == "reset"
        obs = response["obs"]
        return obs, [{}] * self.num_envs

    def call(self, method):
        message = {
            "type": "call",
            "method": method,
        }
        self._send_as_json(message)
        response = self._get_json_dict()

        return response["returns"]

    def close(self):
        message = {
            "type": "close",
        }
        self._send_as_json(message)
        print("close message sent")
        time.sleep(1.0)
        self.connection.close()
        try:
            atexit.unregister(self._close)
        except Exception as e:
            print("exception unregistering close method", e)

    def _close(self):
        print("exit was not clean, using atexit to close env")
        self.close()

    def _launch_env(
        self,
        env_path,
        port,
        show_window,
        framerate,
        seed,
        action_repeat,
        speedup,
        **kwargs,
    ):
        ##############################
        # Convert macOS path if running on Darwin, otherwise use path as-is
        ##############################
        path = convert_macos_path(env_path) if platform == "darwin" else env_path

        launch_cmd = [path]
        launch_cmd.append(f"--port={port}")
        launch_cmd.append(f"--env_seed={seed}")

        ##############################
        # If not showing window, add headless and disable render loop flags
        ##############################
        if show_window is False:
            launch_cmd.append("--disable-render-loop")
            launch_cmd.append("--headless")
        ##############################
        # If framerate is specified, add fixed FPS flag
        ##############################
        if framerate is not None:
            launch_cmd.append(f"--fixed-fps={framerate}")
        ##############################
        # If action repeat is specified, add action repeat flag
        ##############################
        if action_repeat is not None:
            launch_cmd.append(f"--action_repeat={action_repeat}")
        ##############################
        # If speedup is specified, add speedup flag
        ##############################
        if speedup is not None:
            launch_cmd.append(f"--speedup={speedup}")
        ##############################
        # If additional kwargs provided, add them as command line arguments
        ##############################
        if len(kwargs) > 0:
            ##############################
            # Iterate through kwargs and append each as a command line flag
            ##############################
            for key, value in kwargs.items():
                launch_cmd.append(f"--{key}={value}")

        print("DEBUG PY: launch_cmd =", launch_cmd)

        self.proc = subprocess.Popen(
            launch_cmd,
            start_new_session=True,
        )

    def _start_server(self):
        print(f"DEBUG PY: binding 127.0.0.1:{self.port}")
        print(f"waiting for remote GODOT connection on port {self.port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        address = "127.0.0.1"
        server_address = (address, self.port)
        sock.bind(server_address)

        sock.listen(1)
        sock.settimeout(GodotEnv.DEFAULT_TIMEOUT)
        connection, client_address = sock.accept()
        print("DEBUG PY: connection established from", client_address)
        print("connection established")
        return connection

    def _handshake(self):
        message = {
            "type": "handshake",
            "major_version": GodotEnv.MAJOR_VERSION,
            "minor_version": GodotEnv.MINOR_VERSION,
        }

        self._send_as_json(message)

    def _get_env_info(self):
        message = {"type": "env_info"}
        self._send_as_json(message)

        ##############################
        # Loop until receiving env_info message, handling and skipping intermediate acknowledgments
        ##############################
        while True:
            json_dict = self._get_json_dict()
            ##############################
            # If connection closed, raise error
            ##############################
            if json_dict is None:
                raise RuntimeError("Connection to Godot closed while requesting env_info.")

            msg_type = json_dict.get("type")
            ##############################
            # If env_info received, exit loop
            ##############################
            if msg_type == "env_info":
                break

            ##############################
            # If acknowledgment received, continue loop to wait for env_info
            ##############################
            if msg_type == "ack":
                continue

            ##############################
            # If unexpected message type, raise error
            ##############################
            raise AssertionError(f"Unexpected message type '{msg_type}' while waiting for env_info.")

        self.num_envs = json_dict["n_agents"]

        print("action space", json_dict["action_space"])
        ##############################
        # If action space is a single dict, convert it to a list with one entry per agent for compatibility
        ##############################
        if isinstance(json_dict["action_space"], dict):
            json_dict["action_space"] = [json_dict["action_space"]] * self.num_envs

        ##############################
        # Iterate through each agent's action space and build gymnasium action space objects
        ##############################
        for agent_action_space in json_dict["action_space"]:
            tmp_action_spaces = OrderedDict()
            ##############################
            # For each action key-value pair, create appropriate action space object
            ##############################
            for k, v in agent_action_space.items():
                ##############################
                # Create discrete action space with specified size
                ##############################
                if v["action_type"] == "discrete":
                    tmp_action_spaces[k] = spaces.Discrete(v["size"])
                ##############################
                # Create continuous action space with specified size
                ##############################
                elif v["action_type"] == "continuous":
                    tmp_action_spaces[k] = spaces.Box(low=-1.0, high=1.0, shape=(v["size"],))
                ##############################
                # Raise error for unsupported action space types
                ##############################
                else:
                    print(f"action space {v['action_type']} is not supported")
                    assert 0, f"action space {v['action_type']} is not supported"
            tmp_action_spaces_list = list(tmp_action_spaces.items())
            self.action_spaces.append(spaces.Dict(tmp_action_spaces_list))

        print("observation space", json_dict["observation_space"])
        ##############################
        # If observation space is a single dict, convert it to a list with one entry per agent for compatibility
        ##############################
        if isinstance(json_dict["observation_space"], dict):
            json_dict["observation_space"] = [json_dict["observation_space"]] * self.num_envs

        ##############################
        # Iterate through each agent's observation space and build gymnasium observation space objects
        ##############################
        for agent_obs_space in json_dict["observation_space"]:
            observation_spaces = {}
            ##############################
            # For each observation key-value pair, create appropriate observation space object
            ##############################
            for k, v in agent_obs_space.items():
                ##############################
                # Create box observation space (continuous range)
                ##############################
                if v["space"] == "box":
                    ##############################
                    # If observation is 2D, create uint8 box with range 0-255 for image data
                    ##############################
                    if "2d" in k:
                        observation_spaces[k] = spaces.Box(
                            low=0,
                            high=255,
                            shape=v["size"],
                            dtype=np.uint8,
                        )
                    ##############################
                    # Else create float32 box with specified or default range
                    ##############################
                    else:
                        observation_spaces[k] = spaces.Box(
                            low=v.get("low", -1.0),
                            high=v.get("high", 1.0),
                            shape=v["size"],
                            dtype=np.float32,
                        )
                ##############################
                # Create discrete observation space with specified size
                ##############################
                elif v["space"] == "discrete":
                    observation_spaces[k] = spaces.Discrete(v["size"])
                ##############################
                # Raise error for unsupported observation space types
                ##############################
                else:
                    print(f"observation space {v['space']} is not supported")
                    assert 0, f"observation space {v['space']} is not supported"
            self.observation_spaces.append(spaces.Dict(observation_spaces))

        self.agent_policy_names = json_dict.get("agent_policy_names", ["shared_policy"] * self.num_envs)

        return json_dict

    @staticmethod
    def _decode_2d_obs_from_string(
        hex_string,
        shape,
    ):
        return np.frombuffer(bytes.fromhex(hex_string), dtype=np.uint8).reshape(shape)

    def _send_as_json(self, dictionary):
        message_json = json.dumps(dictionary)
        self._send_string(message_json)

    def _get_json_dict(self):
        data = self._get_data()
        return json.loads(data)
    
    def _get_dict_json_message(self):
        data = self._get_data()
        ##############################
        # If no data received, connection to Godot was lost
        ##############################
        if data is None:
            raise RuntimeError("Lost connection to Godot while waiting for message.")
        ##############################
        # Try to parse JSON data; raise error if parsing fails
        ##############################
        try:
            msg = json.loads(data)
        ##############################
        # Catch JSON parsing exception and provide helpful error message
        ##############################
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON from Godot: {data}") from e
        return msg

    def _get_obs(self):
        return self._get_data()

    def _clear_socket(self):
        self.connection.setblocking(False)
        ##############################
        # Try to read and discard all pending data from the socket
        ##############################
        try:
            ##############################
            # Keep receiving data until socket is empty
            ##############################
            while True:
                data = self.connection.recv(4)
                ##############################
                # If no data received, socket is empty so break
                ##############################
                if not data:
                    break
        ##############################
        # Catch exception when no data available in non-blocking mode
        ##############################
        except BlockingIOError:
            pass
        self.connection.setblocking(True)

    def _get_data(self):
        ##############################
        # Try to receive a length-prefixed message from the socket
        ##############################
        try:
            string_size_bytes: bytearray = bytearray()
            received_length: int = 0

            length: int = 4

            ##############################
            # Receive the first 4 bytes containing the message length
            ##############################
            while received_length < length:
                data = self.connection.recv(length - received_length)
                received_length += len(data)
                string_size_bytes.extend(data)

            length = int.from_bytes(string_size_bytes, "little")

            string_bytes: bytearray = bytearray()
            received_length = 0

            ##############################
            # Receive the remaining message data based on received length
            ##############################
            while received_length < length:
                data = self.connection.recv(length - received_length)
                received_length += len(data)
                string_bytes.extend(data)

            string: str = string_bytes.decode()

            return string
        ##############################
        # Catch socket timeout and return None
        ##############################
        except socket.timeout as e:
            print("env timed out", e)
        return None

    def _send_string(self, string):
        message = len(string).to_bytes(4, "little") + bytes(string.encode())
        self.connection.sendall(message)

    def _send_action(self, action):
        self._send_string(action)


def interactive():
    env = GodotEnv()
    print("observation space", env.observation_space)
    print("action space", env.action_space)

    obs = env.reset()
    ##############################
    # Run environment simulation for 1000 steps, sampling actions and stepping environment
    ##############################
    for i in range(1000):
        action = [env.action_space.sample() for _ in range(env.num_envs)]
        action = list(zip(*action))

        obs, reward, term, trunc, info = env.step(action)
    env.close()


if __name__ == "__main__":
    interactive()
