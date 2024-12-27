from thinker.main import make, Env
import thinker
from thinker.actor_net import DRCNet
from torch.utils.data.dataset import Dataset
import torch
from torch.nn.functional import relu
from thinker import util
from typing import Callable, NamedTuple, Optional
from numpy.random import uniform
import os
import argparse
import gym
import numpy as np

def make_current_board_feature_detector(feature_idxs: list, mode: str) -> Callable:
    """Create feature detector functions to extract discrete features from mini-sokoban boards. Boards must be (7,13,13) arrays

    Args:
        feature_idxs (list): list index of feature of interest (see sokoban.cpp);
        mode (str): type of feature detector to construct: "adj" (to count number of adjacent features), "num" (to count total number of features on board) or "loc" (extract location of features)

    Returns:
        Callable: feature detector function, takes in a board state and returns the desired feature
    """
    if mode == "adj":
        def feature_detector(board: torch.tensor) -> int:
            h, w = board.shape[1:]
            x, y = ((board[4,:,:]==1) + (board[5,:,:]==1)).nonzero()[0,:]
            adj_coords = [(xp, yp) for xp, yp in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)] if xp>-1 and xp<h and yp>-1 and yp<w]
            n_hits = 0
            for (xp,yp) in adj_coords:
                for feature_idx in feature_idxs:
                    if board[feature_idx, xp, yp] == 1:
                        n_hits += 1
            return n_hits
    elif mode == "num":
        def feature_detector(board: torch.tensor) -> int:
            return sum([torch.sum((board[feature_idx,:,:]==1).int()) for feature_idx in feature_idxs]).item()
    elif mode == "loc":
        def feature_detector(board):
            locs_xy = sum([(board[feature_idx,:,:]==1) for feature_idx in feature_idxs]).nonzero()
            locs = tuple([(13*x+y).item() for (x,y) in locs_xy]) # each location is an int in range [0,63]
            return locs
    else:
        raise ValueError(f"Please enter a valid mode to construct a feature detector - user entered {mode}, valid modes are adj, num and loc")
    return feature_detector

def make_future_feature_detector(feature_name: str, mode: str, steps_ahead: Optional[int] = None) -> Callable:
    """Create function that adds a feature to each transition (i.e. a dictionary of features) corresponding to the feature with name feature_name in steps_ahead steps

    Args:
        feature_name (str): feature to track steps_ahead into the future
        steps_ahead (Optional int): number of steps ahead into the future to look for this feature if mode is either ahead or traj
        mode (str): type of feature detector to construct: ahead (make feature corresponding to feature_name in steps_ahead steps), traj (make feature corresponding to trajectory of feature_name from current value to over steps_ahead steps) or change (number of steps until the feature next changes)

    Returns:
        Callable: feature detector function, takes in a list of transitions for a single episode and adds an entry for feature_name in steps_ahead steps
    """
    if mode == "ahead":
        new_feature_name = f"{feature_name}_ahead_{steps_ahead}"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry[new_feature_name] = episode_entry[trans_idx+steps_ahead][feature_name] if trans_idx < episode_length-steps_ahead-1 else -1
            return episode_entry
    elif mode == "traj":
        new_feature_name = f"{feature_name}_traj_{steps_ahead}"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                traj = []
                if trans_idx < episode_length-steps_ahead-1:
                    for traj_idx in range(steps_ahead+1):
                        traj.append(episode_entry[trans_idx+traj_idx][feature_name])
                    trans_entry[new_feature_name] = tuple(traj)
                else:
                    trans_entry[new_feature_name] = -1
            return episode_entry
    elif mode == "change":
        new_feature_name = f"{feature_name}_until_change"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            if type(episode_entry[0][feature_name]) == int or type(episode_entry[0][feature_name]) == float:
                for trans_idx, trans_entry in enumerate(episode_entry):
                    future_idx = 0
                    while episode_entry[trans_idx+future_idx][feature_name] == trans_entry[feature_name]:
                        future_idx += 1
                        if trans_idx + future_idx == episode_length: # if no change in feature over rest of episode, just count steps until end of episode (e.g. this is the desired behaviour for counting boxes) - may need to change this if using for features over than boxes
                            break
                    trans_entry[new_feature_name] = future_idx
            elif type(episode_entry[0][feature_name]) == tuple:
                for trans_idx, trans_entry in enumerate(episode_entry):
                    future_idx = 0
                    current_tensor = torch.tensor(episode_entry[trans_idx][feature_name])
                    while trans_idx + future_idx < episode_length - 2:
                        future_idx += 1
                        future_tensor = torch.tensor(episode_entry[trans_idx+future_idx][feature_name])
                        abs_diff = torch.abs(current_tensor - future_tensor)
                        if abs_diff.max().item() > 0:
                            break               
                    trans_entry[new_feature_name] = future_idx
            else:
                raise ValueError(f"Features of type {type(episode_entry[trans_idx])} are not currently supported for {new_feature_name}")
            return episode_entry
    elif mode == "change_loc":
        new_feature_name = f"{feature_name}_change_loc"
        def feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            episode_length = len(episode_entry)
            if type(episode_entry[0][feature_name]) == tuple:
                for trans_idx, trans_entry in enumerate(episode_entry):
                    future_idx = 0
                    current_tensor = torch.tensor(episode_entry[trans_idx][feature_name])
                    change_loc = -1
                    while trans_idx + future_idx < episode_length - 2:
                        future_idx += 1
                        future_tensor = torch.tensor(episode_entry[trans_idx+future_idx][feature_name])
                        abs_diff = torch.abs(current_tensor - future_tensor)
                        if abs_diff.max().item() > 0:
                            change_idx = torch.argmax(abs_diff).item()
                            change_loc = episode_entry[trans_idx][feature_name][change_idx]
                            break
                        elif trans_idx + future_idx == episode_length:
                            change_loc = episode_entry[trans_idx][feature_name][change_idx] # need to check this - are indices of boxes consistent? NO THEY ARE NOT - IGNORE THIS
                            break               
                    trans_entry[new_feature_name] = change_loc  # IS NOW -1 FOR UNCHANGED
            else:
                raise ValueError(f"Features of type {type(episode_entry[trans_idx])} are not currently supported for {new_feature_name}")
            return episode_entry
    else:
        raise ValueError(f"User entered mode {mode}, valid modes are: ahead, traj, change")
    return feature_detector


def make_binary_feature_detector(mode: str, feature_name: str, threshold: int) -> Callable:
    new_feature_name = f"{feature_name}_{mode}_{threshold}"
    if mode == "lessthan":
        def binary_feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            assert type(episode_entry[0][feature_name]) is int or type(episode_entry[0][feature_name]) is float,f"Error: This feature detector constructs binary features from ints or floats, {feature_name} is of type {type(episode_entry[0][feature_name])}"
            episode_length = len(episode_entry)
            for trans_entry in episode_entry:
                trans_entry[new_feature_name] = 1 if (trans_entry[feature_name] <= threshold) else 0
            return episode_entry
    elif mode == "equal":
        def binary_feature_detector(episode_entry: list) -> list:
            assert feature_name in episode_entry[0].keys(), f"Error: This feature detector has been set up to track {feature_name} which is not contained in the episode entry - please re-create it using one of the following features: {episode_entry[0].keys()}"
            assert type(episode_entry[0][feature_name]) is int or type(episode_entry[0][feature_name]) is float,f"Error: This feature detector constructs binary features from ints or floats, {feature_name} is of type {type(episode_entry[0][feature_name])}"
            episode_length = len(episode_entry)
            for trans_entry in episode_entry:
                trans_entry[new_feature_name] = 1 if (trans_entry[feature_name] == threshold) else 0
            return episode_entry
    else:
        raise ValueError("Unsupported mode for make_binary_feature_detector")    
    return binary_feature_detector

def make_trajectory_detector(steps_ahead: int, feature_name: str, mode: str = "default", alt_feature_name: Optional[str] = None, alt_feature_value: Optional[int] = None, inc_current: bool = False) -> Callable:
    #assert feature_name in ["agent_loc", "box_loc", "tracked_box_loc_active", "tar_next", "tracked_box_loc_next", "tar_loc", "next_start_move_box_loc", "next_end_move_box_loc", "box1_loc", "box2_loc", "box3_loc", "box4_loc", "tracked_box_loc_change_after_action_1", "tracked_box_loc_change_after_action_2", "tracked_box_loc_change_after_action_3", "tracked_box_loc_change_after_action_4", "tracked_box_loc_change_with_action_1", "tracked_box_loc_change_with_action_2", "tracked_box_loc_change_with_action_3","tracked_box_loc_change_with_action_4"], "Cannot detect trajectory for feature other than agent or box location"
    if mode == "default":
        def get_future_trajectories(episode_entry: list) -> list:
            #print(feature_name)
            print(len(episode_entry), steps_ahead)
            if feature_name == "agent_loc" or feature_name == "next_start_move_box_loc":
                virtual_ext = [{"agent_loc": episode_entry[-1]["boxnotontar_loc"]}]
            elif feature_name == "tracked_box_loc_active" or feature_name == "tar_next" or feature_name == "tracked_box_loc_next" or feature_name in ["box1_loc","box2_loc", "box3_loc", "box4_loc", "next_end_move_box_loc", "tracked_box_loc_change_after_action_1", "tracked_box_loc_change_after_action_2", "tracked_box_loc_change_after_action_3", "tracked_box_loc_change_after_action_4"]:
                virtual_ext = [{feature_name: episode_entry[-1]["justtar_loc"]}]
            elif feature_name == "spooky_ghost_loc" or feature_name == "nonspooky_ghost_loc" or feature_name == "ghost_loc":
                virtual_ext = [{feature_name: episode_entry[-1][feature_name]}]
            elif feature_name in ["tracked_box_loc_change_with_action_1", "tracked_box_loc_change_with_action_2", "tracked_box_loc_change_with_action_3", "tracked_box_loc_change_with_action_4"]:
                virtual_ext = [{feature_name: tuple()}]
            for trans_idx, trans in enumerate(episode_entry):
                feature_locs_xy = []
                for future_trans in (episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)]:
                    feature_locs_xy += [(future_trans[feature_name][feature_idx] % 13, (future_trans[feature_name][feature_idx]-(future_trans[feature_name][feature_idx]%13))//13) for feature_idx in range(len(future_trans[feature_name]))]
                feature_locs_xy = torch.tensor(feature_locs_xy)
                trajectory = torch.zeros(size=(13,13), dtype=torch.long)
                if len(feature_locs_xy.shape) != 1:
                    trajectory[feature_locs_xy[:,1],feature_locs_xy[:,0]] = 1
                trans[f"{feature_name}_{'future_trajectory' if steps_ahead!=0 else 'current'}_{steps_ahead}"] = trajectory
            return episode_entry
    elif mode == "conjunction":
        def get_future_trajectories(episode_entry: list) -> list:
            if feature_name == "agent_loc":
                virtual_ext = [{"agent_loc": episode_entry[-1]["boxnotontar_loc"]}]
            elif feature_name == "tracked_box_loc_active" or feature_name == "tar_next" or feature_name == "tracked_box_loc_next" or feature_name == "tracked_box_loc_next" or feature_name in ["box1_loc","box2_loc", "box3_loc", "box4_loc"]:
                virtual_ext = [{feature_name: episode_entry[-1]["justtar_loc"]}]
            elif feature_name == "ghost_loc":
                virtual_ext = [{feature_name: episode_entry[-1][feature_name]}]
            virtual_ext[0]["action"] = 0 # dummy no-op action at end of episode
            for trans_idx, trans in enumerate(episode_entry):
                feature_locs_xy = []
                for future_trans in (episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)]:
                    if future_trans[alt_feature_name] == alt_feature_value:
                        feature_locs_xy += [(future_trans[feature_name][feature_idx] % 13, (future_trans[feature_name][feature_idx]-(future_trans[feature_name][feature_idx]%13))//13) for feature_idx in range(len(future_trans[feature_name]))]
                #if type(alt_feature_value) != list:
                    #for future_trans_idx, future_trans in enumerate((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][:-1]):
                        #if future_trans[alt_feature_name] == alt_feature_value:
                            #feature_locs_xy += [((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx] % 13, ((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]-((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]%13))//13) for feature_idx in range(len((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name]))]               
                #else:
                    #for future_trans_idx, future_trans in enumerate((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][:-1]):
                        #for alt_value in alt_feature_value:
                            #if future_trans[alt_feature_name] == alt_value:
                                #feature_locs_xy += [((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx] % 13, ((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]-((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name][feature_idx]%13))//13) for feature_idx in range(len((episode_entry+virtual_ext)[trans_idx+(1 if not inc_current else 0):trans_idx+steps_ahead+(2 if not inc_current else 1)][future_trans_idx+1][feature_name]))]                                   
                feature_locs_xy = torch.tensor(feature_locs_xy)
                trajectory = torch.zeros(size=(13,13), dtype=torch.long)
                if len(feature_locs_xy.shape) != 1:
                    trajectory[feature_locs_xy[:,1],feature_locs_xy[:,0]] = 1
                trans[f"{feature_name}_conj_{alt_feature_name}_equals_{alt_feature_value}_{'future_trajectory' if steps_ahead!=0 else 'current'}_{steps_ahead}"] = trajectory
            return episode_entry
    else:
        raise ValueError(f"Unsupported mode: {mode}")

        
    return get_future_trajectories

def generate_aug_trans(episode_entry):
    trans = episode_entry[-1]
    agent_loc = trans["agent_loc"][0]
    agent_loc = ((agent_loc -(agent_loc % 13))//13, agent_loc % 13,)
    agent_y, agent_x = agent_loc
    box_locs = [((box_loc -(box_loc % 13))//13, box_loc % 13) for box_loc in trans["tracked_box_loc"]]
    wall_locs = [((wall_loc-(wall_loc % 13))//13, wall_loc % 13) for wall_loc in trans["board_state"][0].view(-1).topk(k=(trans["board_state"][0]==1).to(int).sum()).indices]
    if ((agent_y-1,agent_x) in box_locs) and trans["action"] == 1:
        if agent_y > 1 and (agent_y-2,agent_x) not in wall_locs and (agent_y-2,agent_x) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y-1,agent_x) else (agent_y-2,agent_x) for box_loc in box_locs]
            new_agent_loc = (agent_y-1,agent_x)
        else:
            new_agent_loc = agent_loc
            new_box_locs = box_locs
    elif ((agent_y+1,agent_x) in box_locs) and trans["action"] == 2:
        if agent_y < 6 and (agent_y+2,agent_x) not in wall_locs and (agent_y+2,agent_x) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y+1,agent_x) else (agent_y+2,agent_x) for box_loc in box_locs]
            new_agent_loc = (agent_y+1,agent_x)
        else:
            new_agent_loc = agent_loc
            new_box_locs = box_locs
    elif ((agent_y,agent_x-1) in box_locs) and trans["action"] == 3:
        if agent_x > 1  and (agent_y,agent_x-2) not in wall_locs and (agent_y,agent_x-2) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y,agent_x-1) else (agent_y,agent_x-2) for box_loc in box_locs]
            new_agent_loc = (agent_y,agent_x-1)
        else:
            new_box_locs = box_locs
            new_agent_loc = agent_loc
    elif ((agent_y,agent_x+1) in box_locs) and trans["action"] == 4:
        if agent_x < 6 and (agent_y,agent_x+2) not in wall_locs and (agent_y,agent_x+2) not in box_locs:
            new_box_locs = [box_loc if box_loc!=(agent_y,agent_x+1) else ((agent_y,agent_x+2) if agent_x<6 else (agent_y,agent_x+1)) for box_loc in box_locs]
            new_agent_loc = (agent_y,agent_x+1)
        else:
            new_box_locs = box_locs
            new_agent_loc = agent_loc
    else:
        new_box_locs = box_locs
        if trans["action"] == 1 and agent_y > 0:
            new_agent_loc = (agent_y-1,agent_x)
        elif trans["action"] == 2 and agent_y < 7:
            new_agent_loc = (agent_y+1,agent_x)
        elif trans["action"] == 3 and agent_x > 0:
            new_agent_loc = (agent_y,agent_x-1)
        elif trans["action"] == 4 and agent_y < 7:
            new_agent_loc = (agent_y, agent_x+1)
        else:
            new_agent_loc = agent_loc

    new_box_locs = tuple([(13*y+x) for y,x in new_box_locs])
    new_agent_loc = tuple([13*new_agent_loc[0] + new_agent_loc[1]])
    trans = {"agent_loc": new_agent_loc, "action": 0}
    return trans

def make_agent_info_extractor(ahead = 100) -> Callable:
    def agent_info_extractor(episode_entry: list) -> list:
        # track squares from which agent performs actions to leave
        aug_episode_entry = episode_entry #+ [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((13,13), dtype=int)
            cur_level = trans["level"]
            for loc_idx in range(169):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx:-1][:ahead]):
                    if loc_idx in future_trans["agent_loc"] and (future_trans["agent_loc"] != aug_episode_entry[trans_idx+future_trans_idx+1]["agent_loc"]): #NB: ignore no-ops and effective no-ops since want action we leave square with 
                        board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] = future_trans["action"]
                        break
                    elif future_trans["level"] != cur_level:
                        break
            trans[f"agent_onto_with_{ahead}"] = board_locs
            new_board_locs = torch.zeros((13,13), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans[f"agent_from_{ahead}"] = new_board_locs
        episode_entry = aug_episode_entry
        # track squares from which agent performs action to enter
        aug_episode_entry = episode_entry #+ [generate_aug_trans(episode_entry)]
        for trans_idx, trans in enumerate(aug_episode_entry):
            board_locs = torch.zeros((13,13), dtype=int)
            cur_level = trans["level"]
            for loc_idx in range(169):
                for future_trans_idx, future_trans in enumerate(aug_episode_entry[trans_idx+1:][:ahead]):
                    if loc_idx in future_trans["agent_loc"] and aug_episode_entry[trans_idx+future_trans_idx]["agent_loc"] != future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] = aug_episode_entry[trans_idx+future_trans_idx]["action"]
                        break
                    elif future_trans["level"] != cur_level:
                        break
            trans[f"agent_onto_after_{ahead}"] = board_locs
            new_board_locs = torch.zeros((13,13), dtype=int)
            new_board_locs[board_locs != 0 ] = 1
            trans[f"agent_onto_{ahead}"] = new_board_locs
        episode_entry = aug_episode_entry
        for trans_idx, trans in enumerate(episode_entry):
            board_locs = torch.zeros((13,13), dtype=int)
            cur_level = trans["level"]
            for loc_idx in range(169):
                for future_trans in episode_entry[trans_idx+1:][:ahead]:
                    if loc_idx in future_trans["agent_loc"]:
                        board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] += (1 if board_locs[(loc_idx-loc_idx%13)//13, loc_idx%13] <= 2 else 0)
                    elif future_trans["level"] != cur_level:
                        break
            trans["agent_loc_count"] = board_locs
        return episode_entry
    return agent_info_extractor



def make_bfs_feature_detector(max_depth: int, board_size: int = 13) -> Callable:
    """
    Create a BFS feature detector to generate probing targets.

    Args:
        max_depth (int): Maximum depth for the BFS.
        board_size (int): Size of the board (default is 13x13 for Mini Pacman).

    Returns:
        Callable: A feature detector function to add BFS-based features to transitions.
    """
    from collections import deque

    def bfs_with_distance(board_state, start_pos, depth, visited):
        """
        Perform BFS to calculate reachable squares within the given depth.

        Args:
            board_state: The grid representation of the board (walls as 1, free spaces as 0).
            start_pos: Starting position of the agent (x, y).
            depth: Maximum depth to explore.
            visited: Matrix to mark visited positions.
        """
        queue = deque([(start_pos, 0)])  # Queue contains ((x, y), current_depth)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # [Up, Down, Left, Right]

        while queue:
            (x, y), current_depth = queue.popleft()

            if current_depth > depth or visited[y, x]:
                continue

            visited[y, x] = 1  # Mark the square as visited

            # Explore neighbors
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < board_size and 0 <= ny < board_size and board_state[ny, nx] == 0:
                    queue.append(((nx, ny), current_depth + 1))

    def bfs_feature_detector(episode_entry: list) -> list:
        """
        Add BFS features to the transitions in an episode entry.

        Args:
            episode_entry: List of transitions in an episode.

        Returns:
            Updated episode entry with BFS features added.
        """
        for trans_idx, trans_entry in enumerate(episode_entry):
            agent_loc = trans_entry["agent_loc"][0]
            agent_x, agent_y = agent_loc % board_size, agent_loc // board_size

            # Extract board state and create a visited matrix for each depth
            board_state = (trans_entry["board_state"][0] == 1).int()  # Wall cells as 1, others as 0
            for depth in range(1, max_depth + 1):
                visited = torch.zeros((board_size, board_size), dtype=torch.float32)
                bfs_with_distance(board_state, (agent_x, agent_y), depth, visited)
                visited[agent_y, agent_x] = 0  # Exclude the agent's current location
                trans_entry[f"bfs_depth_{depth}"] = visited

        return episode_entry

    return bfs_feature_detector


@torch.no_grad()
def create_probing_data(drc_net: DRCNet, env: Env, flags: NamedTuple, num_episodes: int, current_board_feature_fncs: list, future_feature_fncs: list,
                         binary_feature_fncs: list, prob_accept: float = 1.0, debug: bool = False, device: torch.device = torch.device("cpu")) -> list:
    """Generate a list where each entry is a dictionary of features corresponding to a single transition

    Args:
        drc_net (DRCNet): Trained DRC network used to generate transitions
        env (Env): Sokoban environment
        flags (NamedTuple): flag object
        num_episodes (int): number of episodes to run to generate the transitions
        current_board_feature_fncs (list): list of tuples of the form (feature_name, feature_fnc), where each feature_fnc extracts a discrete feature from the current state of the Sokoban board; this feature is then added to the episode entry (dictionary) with the key feature_name
        future_feature_fncs (list): list of functions where each function adds a feature to the current transition corresponding to the value taken by some other feature in a future transition
        prob_accept (float): probability that each transition entry is independently accepted into the dataset

    Returns:
        list: returns probing_data, a list of dictionaries where each dictionary contains features for a single transition generated by the DRC agent
    """
    

    rnn_state = drc_net.initial_state(batch_size=1, device=device)
    state = env.reset()
    state = {"real_states": torch.tensor(state).permute(2,0,1).unsqueeze(0)}
    env_out = util.init_env_out(state, flags, dim_actions=1, tuple_action=False)

    episode_length = 0
    board_num = 0
    probing_data = []
    episode_entry = []

    actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
    trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
    trans_entry["action"] = actor_out.action.item()
    trans_entry["value"] = round(actor_out.baseline.item(), 3) 
    trans_entry["board_state"] = state["real_states"][0].detach().cpu() # tensor of size (channels, board_height, board_width)
    trans_entry["hidden_states"] = drc_net.hidden_state[0].detach().cpu() # tensor of size (ticks+1, layers*169, representation_height, representation_width)
    trans_entry["board_num"] = board_num
    trans_entry["level"] = env.level
    episode_length += 1

    while(board_num < num_episodes):

        state, reward, done, info = env.step(actor_out.action)
        state = {"real_states": torch.tensor(state).permute(2,0,1).unsqueeze(0)}
        trans_entry["reward"] = round(reward, 3) # round rewards to 3 d.p.
        episode_entry.append(trans_entry)
        print(episode_length, reward, actor_out.action)

        if done:
            for fnc in future_feature_fncs:
                print(len(episode_entry))
                episode_entry = fnc(episode_entry)
            for trans_idx, trans_entry in enumerate(episode_entry):
                trans_entry["steps_remaining"] = episode_length - trans_idx
                trans_entry["steps_taken"] = trans_idx+1
                trans_entry["return"] = sum([(0.97**t)*future_trans["reward"] for t, future_trans in enumerate(episode_entry[trans_idx:])])

            for fnc in binary_feature_fncs:
                episode_entry = fnc(episode_entry)
            
            
            probing_data += episode_entry 
            
            episode_length = 0
            board_num += 1
            print("Data collected from episode", board_num, "with episode length of", len(episode_entry))
            episode_entry = []
            rnn_state = drc_net.initial_state(batch_size=1, device=device)
            state = env.reset()
            state = {"real_states": torch.tensor(state).permute(2,0,1).unsqueeze(0)}

        env_out = util.create_env_out(actor_out.action, state, torch.tensor([reward]), torch.tensor([done]), info, flags)
        actor_out, rnn_state = drc_net(env_out, rnn_state, greedy=True)
        if debug:
            print(actor_out.pri_param.argmax(dim=-1).item(), actor_out.action.item())

        trans_entry = {feature:fnc(state["real_states"][0]) for feature, fnc in current_board_feature_fncs}
        trans_entry["action"] = actor_out.action.item()
        trans_entry["value"] = round(actor_out.baseline.item(), 3) 
        trans_entry["board_state"] = state["real_states"][0].detach().cpu() # tensor of size (channels, board_height, board_width)
        trans_entry["hidden_states"] = drc_net.hidden_state[0].detach().cpu() # tensor of size (ticks+1, layers*169, representation_height, representation_width)
        trans_entry["board_num"] = board_num
        trans_entry["level"] = env.level
        episode_length += 1

    return probing_data

def make_selector(mode: str, feature_name: str = "agent_loc", threshold: int = 1, prob_accept: float = 1.0) -> list:
    if mode == "random":
        def selector(probing_data: list) -> list:
            pruned_data = []
            for trans_entry in probing_data:
                if prob_accept > uniform(0,1):
                    pruned_data.append(trans_entry)
            return pruned_data
    elif mode == "lessthan":
        def selector(probing_data: list) -> list:
            pruned_data = []
            for trans_entry in probing_data:
                if trans_entry[feature_name] <= threshold and prob_accept > uniform(0,1):
                    pruned_data.append(trans_entry)
            return pruned_data
    elif mode == "greaterthan":
        def selector(probing_data: list) -> list:
            pruned_data = []
            for trans_entry in probing_data:
                if trans_entry[feature_name] >= threshold and prob_accept > uniform(0,1):
                    pruned_data.append(trans_entry)
            return pruned_data
    else:
        raise ValueError(f"no such mode as {mode} supported by make_selector")
    return selector

class ProbingDataset(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index: int) -> dict:
        return self.data[index]
    def get_feature_range(self, feature: str) -> tuple[int, int]:
        assert feature in self.data[0].keys(), f"Please enter a feature in dataset: {self.data[0].keys()}"
        min_feature_value, max_feature_value = self.data[0][feature], self.data[0][feature]
        for entry in self.data:
            if entry[feature] > max_feature_value:
                max_feature_value = entry[feature]
            elif entry[feature] < min_feature_value:
                min_feature_value = entry[feature]
        return (min_feature_value, max_feature_value)


class ProbingDatasetCleaned(Dataset):
    def __init__(self, data: list):
        self.data = data
    def __len__(self) -> int:
        return len(self.data)
    def __getitem__(self, index: int) -> tuple:
        return self.data[index]
        

if __name__=="__main__":
    import gym_pilleater

    parser = argparse.ArgumentParser(description="run convprobe patching exps")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--model_name", type=str, default="250m")
    parser.add_argument("--env_name", type=str, default="")
    parser.add_argument("--unq", type=bool, default=False)
    parser.add_argument("--pct_train", type=float, default=1)
    parser.add_argument("--gpu", type=bool, default=False)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--mini", type=bool, default=True)
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_ticks", type=int, default=3)
    args = parser.parse_args()

    mini = args.mini
    name = args.name
    gpu = args.gpu
    pct_train = args.pct_train
    unq = args.unq
    num_episodes = args.num_episodes
    debug = args.debug
    model_name = args.model_name
    env_name = args.env_name

    
   
    
    env = thinker.make(
                f"gym_pilleater/PillEater-v0", 
                env_n=1, 
                gpu=False,
                wrapper_type=1, 
                has_model=False, 
                train_model=False, 
                parallel=False, 
                save_flags=False,
                mini=True,
                mini_unqtar=False,
                mini_unqbox=False         
            ) 
    flags = util.create_setting(args=[], save_flags=False, wrapper_type=1) 
    flags.mini = True
    flags.mini_unqtar = False
    flags.mini_unqbox = False
    drc_net = DRCNet(
        obs_space=env.observation_space,
        action_space=env.action_space,
        flags=flags,
        record_state=True,
        input_dim=14,
        num_ticks=args.num_ticks,
        num_layers=args.num_layers
    )
    ckp_path = "../../checkpoints/pilleater"
    #ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep{model_name}.tar")
    ckp_path = os.path.join(util.full_path(ckp_path), f"ckp_actor_realstep250m.tar")
    ckp = torch.load(ckp_path, map_location=torch.device('cpu'))
    drc_net.load_state_dict(ckp["actor_net_state_dict"], strict=False)
    env = gym.make("gym_pilleater/PillEater-v0")
    env.seed(args.seed)

    if gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    agent_loc_detector = make_current_board_feature_detector(feature_idxs=[13], mode="loc")
    spooky_ghost_loc_detector = make_current_board_feature_detector(feature_idxs=[2,6,10], mode="loc")
    nonspooky_ghost_loc_detector = make_current_board_feature_detector(feature_idxs=[3,4,7,8,11,12], mode="loc")
    ghost_loc_detector = make_current_board_feature_detector(feature_idxs=[2,3,4,6,7,8,10,11,12], mode="loc")

    current_board_feature_fncs = [("agent_loc", agent_loc_detector),
                                  ("ghost_loc", ghost_loc_detector),
                                  ("spooky_ghost_loc", spooky_ghost_loc_detector),
                                  ("nonspooky_ghost_loc", nonspooky_ghost_loc_detector)]
    future_feature_fncs = [
                        make_agent_info_extractor(ahead=3),
                        make_agent_info_extractor(ahead=6),
                        make_agent_info_extractor(ahead=8),
                        make_agent_info_extractor(ahead=10),
                        make_agent_info_extractor(ahead=12),
                        make_agent_info_extractor(ahead=600),
                        make_agent_info_extractor(ahead=16),
                        ]
    future_feature_fncs += [make_trajectory_detector(feature_name="ghost_loc", steps_ahead=i) for i in [1,3,6]]
    future_feature_fncs += [make_trajectory_detector(feature_name="spooky_ghost_loc", steps_ahead=i) for i in [1,3,6]]
    future_feature_fncs += [make_trajectory_detector(feature_name="nonspooky_ghost_loc", steps_ahead=i) for i in [1,3,6]]

    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=i, mode="conjunction", alt_feature_name="action", alt_feature_value=j) for i in [120] for j in [1,2,3,4]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=i, mode="conjunction", alt_feature_name="action", alt_feature_value=[j,k]) for i in [120] for j in [1,2,3,4] for k in [1,2,3,4] if j>k]
    #future_feature_fncs += [make_trajectory_detector(feature_name="box_loc", steps_ahead=i) for i in [1,5,10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tracked_box_loc_active", steps_ahead=i) for i in [1,5,10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name=f"tracked_box_loc_change_after_action_{j}", steps_ahead=i) for i in [10,20,120] for j in [1,2,3,4]]
    #future_feature_fncs += [make_trajectory_detector(feature_name=f"tracked_box_loc_change_with_action_{j}", steps_ahead=i) for i in [10,20,120] for j in [1,2,3,4]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tracked_box_loc_next", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tar_next", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tar_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="box_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="agent_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="tracked_box_loc_active", steps_ahead=0, inc_current=True)]

    #future_feature_fncs += [make_trajectory_detector(feature_name="next_end_move_box_loc", steps_ahead=0, inc_current=True)]
    #future_feature_fncs += [make_trajectory_detector(feature_name="next_end_move_box_loc", steps_ahead=i) for i in [10,20,120]]
    #future_feature_fncs += [make_trajectory_detector(feature_name="next_start_move_box_loc", steps_ahead=0, inc_current=True)]

    #if unq:
        #future_feature_fncs += [make_trajectory_detector(feature_name=f"box{j}_loc", steps_ahead=i) for i in [1,5,10,20,120] for j in [1,2,3,4]] 
    
    


    probing_data = create_probing_data(drc_net=drc_net,
                                       env=env,
                                       flags=flags,
                                       num_episodes=num_episodes,
                                       current_board_feature_fncs=current_board_feature_fncs,
                                       future_feature_fncs=future_feature_fncs,
                                       binary_feature_fncs=[], 
                                       debug=debug,
                                       device=device)
    
    if debug:
        for trans in probing_data[:250]: # check that h,c,x_enc correctly ordered by ensuring decoding with policy head behaves as expected
            x = trans["hidden_states"].to(env.device)
            core_output = x[-1,169*2:169*2+32,:,:]
            x_enc = x[-1,192:224,:,:]
            core_output = torch.cat([x_enc, core_output], dim=0)
            core_output = torch.flatten(core_output).view(1,-1)
            final_out = relu(drc_net.final_layer(core_output))
            pri_logits = drc_net.policy(final_out)
            assert torch.argmax(pri_logits, dim=-1).item() == trans["action"], "hidden states are incorrectly ordered - decoding [h,x_enc] with the policy head does not produce the chosen action as expected"

    final_train_board = int(num_episodes * pct_train)
    final_val_board = final_train_board + int(num_episodes * round(0.5 * (1 - pct_train), 2))
    probing_train_data = [entry for entry in probing_data if entry["board_num"] <= final_train_board]
    probing_val_data = [entry for entry in probing_data if entry["board_num"] > final_train_board and entry["board_num"] <= final_val_board]
    probing_test_data = [entry for entry in probing_data if entry["board_num"] > final_val_board]
    
    print(f"Full train, val and test sets contain {len(probing_train_data)}, {len(probing_val_data)}, {len(probing_test_data)} transitions respectively")
    
    if not debug:
        torch.save(ProbingDataset(probing_train_data), f"./data/{name}_data_full_{model_name}.pt")
        #torch.save(ProbingDataset(probing_val_data), f"./data/val_data_full_{model_name}.pt")
        #torch.save(ProbingDataset(probing_test_data), f"./data/test_data_full_{model_name}.pt")

    selectors = [
            ]
        #("adjbox", make_selector(mode="greaterthan", feature_name="adj_box", threshold=1, prob_accept=0.2)),
        #("noadjbox", make_selector(mode="lessthan", feature_name="adj_box", threshold=0, prob_accept=0.2)),
        #("soon5", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=5, prob_accept=0.3)),
        #("soon4", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=4, prob_accept=0.3)),
        #("soon3", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=3, prob_accept=1)),
        #("soon2", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=2, prob_accept=1)),
        #("soon1", make_selector(mode="lessthan", feature_name="num_boxnotontar_until_change", threshold=1, prob_accept=1)),
        #("start3", make_selector(mode="lessthan", feature_name="steps_taken", threshold=3)),
        #("start2", make_selector(mode="lessthan", feature_name="steps_taken", threshold=2)),
        #("start1", make_selector(mode="lessthan", feature_name="steps_taken", threshold=1)),
        #("onebox", make_selector(mode="lessthan", feature_name="num_boxnotontar", threshold=1, prob_accept=0.2))

