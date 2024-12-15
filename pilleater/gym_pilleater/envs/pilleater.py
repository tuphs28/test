import numpy as np
import gym
from gym import spaces
import math
import random
from heapq import heappush, heappop

class PillEater:
    WALLS = 0

    FOOD = 1
    GHOSTONFOOD = 2
    GHOSTONFOOD_EDIBLE = 3
    GHOSTONFOOD_EDIBLE_END = 4

    NOTFOOD = 5
    GHOSTNOTONFOOD = 6
    GHOSTNOTONFOOD_EDIBLE = 7
    GHOSTNOTONFOOD_EDIBLE_END = 8

    PILL = 9
    GHOSTONPILL = 10
    GHOSTONPILL_EDIBLE = 11
    GHOSTONPILL_EDIBLE_END = 12
    
    PILLMAN = 13

STANDARD_MAP = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]


class PillEaterEnv(gym.Env):

    def __init__(self, mini=True, mini_unqtar=False, mini_unqbox=False, dan_num=0, shape = 13):

        self.shape = shape
        #self.map_basic = np.array(STANDARD_MAP, dtype=np.uint8)
        #self.height, self.width = self.map_basic.shape
        self.nplanes = 14  
        self.image = np.zeros((self.shape, self.shape, self.nplanes), dtype=np.uint8)
        
        self.pillman_pos = None 
        self.ghosts_pos = None
        self.food = None
        self.pills = None

        self.nghosts_init = 4
        self.ghost_speed_init = 0.5
        self.ghost_speed = self.ghost_speed_init
        self.ghost_speed_increase = 0.1
        self.stochasticity = 0.05
        self.safe_distance = 5

        self.g_lam = 1
        self.g_unif = 2

        self.npills = 4
        self.pill_duration = 10

        self.dir_vec = np.array([
            [0, 0],     # no-op
            [-1, 0],    # up 
            [1, 0],     # down
            [0, -1],    # left
            [0, 1]      # right
        ])
        self.reverse_dir = (0, 2, 1, 4, 3)
        self.nactions = len(self.reverse_dir)

        self.step_reward = 0
        self.food_reward = 1
        self.big_pill_reward = 2
        self.ghost_hunt_reward = 5
        self.ghost_death_reward = 0
        self.all_pill_terminate = False
        self.all_ghosts_terminate = False
        self.all_food_terminate = True
        self.timer_terminate = -1
        self.discount = 1
        self.frame_cap = 500
    
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.shape, self.shape, self.nplanes), dtype=np.uint8)
        self.action_space = spaces.Discrete(self.nactions)
        
        
    def make_movement_arrays(self, map_array):
        """Create movement arrays corresponding to the change in location following actions in grid squares, and corresponding to the wall locations."""

        walls = (map_array == 1).astype(np.uint8)

        height, width = map_array.shape
        movement_map = np.zeros((height, width, self.nactions, 2), dtype=np.int32)
        for y in range(height):
            for x in range(width):
                for action in range(self.nactions):
                    if y == 0 and action == 1:
                        movement_map[y, x, action] = [y, x]
                    elif y == height - 1 and action == 2:
                        movement_map[y, x, action] = [y, x]
                    elif x == 0 and action == 3:
                        movement_map[y, x, action] = [y, x]
                    elif x == width - 1 and action == 4:
                        movement_map[y, x, action] = [y, x]
                    else:
                        ny, nx = y + self.dir_vec[action][0], x + self.dir_vec[action][1]
                        if walls[ny, nx] == 0:
                            movement_map[y, x, action] = [ny, nx]
                        else:
                            movement_map[y, x, action] = [y, x]

        return movement_map, walls
    
    def reset(self, options=None, room_id=None):
        """Initialize a new episode."""
        self.frame = 0
        self.ghost_speed = self.ghost_speed_init
        self.timer = 0
        self.pcontinue = 1
        self.reward = 0

        #self.nghosts_init = 1 #+ np.random.poisson(lam=self.g_lam)

        if room_id:
            self.level = room_id
        else:
            self.level = 1

        self.nghosts_init = 1 + np.random.poisson(lam=1)
      
        self._init_level(self.level)
        observation = self._make_image()

        return observation
    
    def step(self, action):
        """Move environment one time-step forward after Pillman performs action."""
        self.frame += 1
        self.pcontinue = self.discount
        self.reward = self.step_reward
        self.timer += 1
        self.world_state["power"] = max(0, self.world_state["power"] - 1)
        ghost_speed_modifier = 0.5 if self.world_state["power"] > 0 else 1

        # Move Pillman
        self._move_pillman(action)

        for i, ghost in enumerate(self.world_state["ghosts"]):
            if np.array_equal(ghost["pos"], self.world_state["pillman"]["pos"]):
                if self.world_state["power"] == 0:
                    self._die_by_ghost()
                else:
                    self._kill_ghost(i)
                break 

        # Move ghosts
        for i, ghost in enumerate(self.world_state["ghosts"]):
            speed = self.ghost_speed * ghost_speed_modifier
            if np.random.uniform() < speed:
                self._move_ghost(ghost, i)
                if np.array_equal(ghost["pos"], self.world_state["pillman"]["pos"]):
                    if self.world_state["power"] == 0:
                        self._die_by_ghost()
                    else:
                        self._kill_ghost(i)
                        break 

        # Check if move to next level or end
        if self.nfood == 0:
            self._init_level(self.level + 1)
        if self.frame >= self.frame_cap:
            self.pcontinue = 0

        observation = self._make_image()
        done = self.pcontinue == 0
        info = {}
        return observation, self.reward, done, info
    
    def get_random_position(self, map_array):
        """Get a random available position"""
        zeros = np.argwhere(map_array == 0)
        idx = np.random.randint(zeros.shape[0])
        return zeros[idx]
    
    def _make_pillman(self):
        """Initalize Pillman at a random position."""
        pos = self.get_random_position(self.walls)
        return {"pos": pos}
    

    def _make_enemy(self):
        """Initialize a ghost at a random position >= self.safe_distance away from Pillman"""
        occupied_map = self.walls.copy()
        y_pillman, x_pillman = self.world_state["pillman"]["pos"]
        y_indices, x_indices = np.ogrid[:self.shape, :self.shape]
        distance_squared = (y_indices - y_pillman) ** 2 + (x_indices - x_pillman) ** 2
        mask = (distance_squared >= self.safe_distance ** 2)
        available_positions = np.argwhere((occupied_map == 0) & mask)
        idx = np.random.randint(available_positions.shape[0])
        pos = available_positions[idx]
        dir = np.random.randint(4)
        return {"pos": pos, "dir": dir}
    
    def _make_pill(self):
        """Initialize a pill at a random position"""
        pos = self.get_random_position(self.walls)
        return {"pos": pos}
    
    def _init_level(self, level):
        """Initializes a new level"""
        self.level = level
        self.frame = 0
        self.map_basic = self.generate_new_maze(self.shape)
        self.map, self.walls = self.make_movement_arrays(self.map_basic)

        self.world_state = {
            "pillman": self._make_pillman(),
            "ghosts": [],
            "food": np.zeros((self.shape, self.shape), dtype=np.uint8),
            "notfood": np.zeros((self.shape, self.shape), dtype=np.uint8),
            "pills": [],
            "power": 0
        }

        self.world_state["food"] = (self.walls == 0).astype(np.uint8)
        self.world_state["notfood"] = np.zeros_like(self.world_state["food"])
        y_pillman, x_pillman = self.world_state["pillman"]["pos"]
        self.world_state["food"][y_pillman, x_pillman] = 0
        self.world_state["notfood"][y_pillman, x_pillman] = 1
        self.nfood = np.sum(self.world_state["food"])

        self.world_state["pills"] = [self._make_pill() for _ in range(self.npills)]

        self.nghosts = int(np.floor(self.nghosts_init + (0.25+np.random.uniform(low=0, high=self.g_unif))*(level-1)))
        #self.nghosts = int(self.nghosts_init + math.floor((level - 1)))
        self.world_state["ghosts"] = [self._make_enemy() for _ in range(self.nghosts)]
        self.world_state["power"] = 0
        self.ghost_speed = self.ghost_speed_init + self.ghost_speed_increase * (level - 1)
        self.timer = 0

    def _get_food(self, y, x):
        """Remove food from the tile Pillman just entered"""
        self.reward += self.food_reward
        self.world_state["food"][y, x] = 0
        self.world_state["notfood"][y, x] = 1
        self.nfood -= 1

    def _get_pill(self, pill_index):
        """Consume pill from tile Pillman just entered"""
        pill = self.world_state["pills"].pop(pill_index)
        self.reward += self.big_pill_reward
        self.world_state["power"] = self.pill_duration

    def _kill_ghost(self, ghost_index):
        """Pillman kills a ghost."""
        self.world_state["ghosts"].pop(ghost_index)
        self.reward += self.ghost_hunt_reward

    def _die_by_ghost(self):
        """Pillman is killed by a ghost."""
        self.reward += self.ghost_death_reward
        self.pcontinue = 0

    def _move_pillman(self, action):
        """Pillman performs the action."""
        pos = self.world_state["pillman"]["pos"]
        new_pos = self.map[pos[0], pos[1], action]
        self.world_state["pillman"]["pos"] = new_pos

        y, x = new_pos
        if self.world_state["food"][y, x] == 1:
            self._get_food(y, x)

        for i, pill in enumerate(self.world_state["pills"]):
            if np.array_equal(pill["pos"], new_pos):
                self._get_pill(i)
                break

    def _move_ghost(self, ghost, ghost_index):
        """Moves the ghost using A* pathfinding for chasing or fleeing, ensuring no collisions."""
        def heuristic(pos, target, fleeing=False):
            """Heuristic function: Manhattan distance."""
            dist = abs(pos[0] - target[0]) + abs(pos[1] - target[1])
            return -dist if fleeing else dist  # Invert distance if fleeing

        def neighbors(pos):
            """Get valid neighbors of the current position."""
            y, x = pos
            valid_moves = []
            for action in range(1, self.nactions):
                new_pos = tuple(self.map[y, x, action])
                if not np.array_equal(new_pos, pos):  # Ensure it's a valid move
                    valid_moves.append((new_pos, action))
            return valid_moves

        # Define the start and goal positions
        start = tuple(ghost["pos"])
        goal = tuple(self.world_state["pillman"]["pos"])
        fleeing = self.world_state["power"] > 0  # Determine if the ghost should flee

        # if not fleeing, do epsilon-greedy chase
        if not fleeing and self.stochasticity > np.random.uniform():
            self._move_ghost_random(ghost, ghost_index)
            return

        # If the ghost is already at the goal, no movement is needed (edge case)
        if start == goal and not fleeing:
            return

        # A* algorithm setup
        open_set = []
        heappush(open_set, (heuristic(start, goal, fleeing), 0, start, None))  # (f-score, g-score, position, action)
        came_from = {}
        g_score = {start: 0}
        action_map = {}

        # Perform A* search
        while open_set:
            _, current_g, current, action = heappop(open_set)

            # If the goal is reached (chasing mode only), reconstruct the path
            if current == goal and not fleeing:
                while action_map[current] != start:
                    current = action_map[current]
                next_action = came_from[current]
                new_pos = self.map[start[0], start[1], next_action]
                if not self._is_ghost_collision(new_pos, ghost_index):
                    ghost["dir"] = next_action
                    ghost["pos"] = new_pos
                return

            for neighbor, neighbor_action in neighbors(current):
                tentative_g = current_g + 1  # Each move costs 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal, fleeing)
                    heappush(open_set, (f_score, tentative_g, neighbor, neighbor_action))
                    came_from[neighbor] = neighbor_action
                    action_map[neighbor] = current

        # If no path is found, fall back to random movement
        self._move_ghost_random(ghost, ghost_index)

    def _is_ghost_collision(self, pos, ghost_index):
        """Check if the given position collides with any other ghost."""
        for i, other_ghost in enumerate(self.world_state["ghosts"]):
            if i != ghost_index and np.array_equal(pos, other_ghost["pos"]):
                return True
        return False

    def _move_ghost_random(self, ghost, ghost_index):
        """Fallback to random movement if A* fails (e.g., no valid path)."""
        pos = ghost["pos"]
        available_moves = []
        for i in range(1, self.nactions):
            new_pos = self.map[pos[0], pos[1], i]
            if not np.array_equal(new_pos, pos) and not self._is_ghost_collision(new_pos, ghost_index):
                available_moves.append(i)
        if available_moves:
            ghost["dir"] = np.random.choice(available_moves)
            ghost["pos"] = self.map[pos[0], pos[1], ghost["dir"]]


    def _move_all_ghosts(self):
        """Moves all ghosts while ensuring no two occupy the same square."""
        for i, ghost in enumerate(self.world_state["ghosts"]):
            self._move_ghost(ghost, i)


    def _make_image(self):
        """Create the symbolic observation"""
        self.image.fill(0)
        self.image[:, :, PillEater.WALLS] = self.walls
        self.image[:, :, PillEater.FOOD] = self.world_state["food"]
        self.image[:, :, PillEater.NOTFOOD] = self.world_state["notfood"]

        y_pillman, x_pillman = self.world_state["pillman"]["pos"]
        self.image[y_pillman, x_pillman, PillEater.PILLMAN] = 1 

        if self.world_state["pills"]:
            pill_positions = np.array([pill["pos"] for pill in self.world_state["pills"]])
            y_pill, x_pill = pill_positions[:, 0], pill_positions[:, 1]
            self.image[y_pill, x_pill, PillEater.PILL] = 1
            self.image[y_pill, x_pill, PillEater.FOOD] = 0

        edibility = self.world_state['power'] / float(self.pill_duration)
        if edibility > 0.2:
            ghost_idx = 3
        elif edibility > 0:
            ghost_idx = 2
        else:
            ghost_idx = 1

        for ghost in self.world_state["ghosts"]:
            y_ghost, x_ghost = ghost["pos"]
            if self.image[y_ghost, x_ghost, PillEater.FOOD] == 1:
                self.image[y_ghost, x_ghost, PillEater.FOOD] = 0
                self.image[y_ghost, x_ghost, PillEater.FOOD+ghost_idx] = 1
            elif self.image[y_ghost, x_ghost, PillEater.NOTFOOD] == 1:
                self.image[y_ghost, x_ghost, PillEater.NOTFOOD] = 0
                self.image[y_ghost, x_ghost, PillEater.NOTFOOD+ghost_idx] = 1
            elif self.image[y_ghost, x_ghost, PillEater.PILL] == 1:
                self.image[y_ghost, x_ghost, PillEater.PILL] = 0
                self.image[y_ghost, x_ghost, PillEater.PILL+ghost_idx] = 1

        return self.image

    def seed(self, seed):
        """Sets the random seed."""
        self.seed_value = seed
        np.random.seed(seed)

    def generate_new_maze(self, size = 13, p = 0.15):
        grid = np.ones((size, size), dtype=int)
        
        start_x, start_y = random.randrange(0, size, 2), random.randrange(0, size, 2)
        grid[start_x, start_y] = 0
        
        frontiers = set()
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == 1:
                frontiers.add((nx, ny, start_x, start_y))
        
        while frontiers:
            fx, fy, px, py = random.choice(list(frontiers))
            frontiers.remove((fx, fy, px, py))
            
            if grid[fx, fy] == 1:
                grid[fx, fy] = 0
                grid[(fx + px) // 2, (fy + py) // 2] = 0

                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    nx, ny = fx + dx, fy + dy
                    if 0 <= nx < size and 0 <= ny < size and grid[nx, ny] == 1:
                        frontiers.add((nx, ny, fx, fy))
        
        for x in range(1, size - 1):
            for y in range(1, size - 1):
                if grid[x, y] == 1 and random.random() < p:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        if grid[x + dx, y + dy] == 0:
                            grid[x, y] = 0
                            break
                    
        return grid