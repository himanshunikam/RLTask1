from config import *
import numpy as np
import random

# 'S': start
# 'G': goal
# ' ': empty cell
# '#': mountain
# 'O': water
# 'A': airport (only two allowed in total)

class GridworldEnv():
    def __init__(self):
        """
        Initialize Gridworld
        """
        self.reset()


    def shape(self):
        """
        Returns the shape (number of rows and columns) of the Gridworld environment
        :return: (number of rows, number of columns)
        """
        return len(self.grid), len(self.grid[0])


    def num_states(self):
        """
        Returns the number of states of the Gridworld environment
        :return: number of states (rows times columns)
        """
        return len(self.grid) * len(self.grid[0])


    def num_actions(self):
        """
        Returns the number of possible actions (UP/RIGHT/DOWN/LEFT)
        :return:
        """
        return 4


    def step(self, action):
        """
        Execute a step with the agent in the Gridworld environment. Perform an action and obtain an observation,
        a reward and the information (done) whether the agent ended up in a terminal state. Do not use this function
        for value iteration / policy iteration but the function below.
        :param action: Scalar value of the action (UP/RIGHT/DOWN/LEFT) according to config file
        :return: (state, reward, done flag)
        """
        # use this function normally
        self.state = self._calc_next_state(action)
        reward = self._calc_reward()
        done = self._calc_done()
        obs = self._state_to_obs(self.state)
        return obs, reward, done


    def step_dp(self, obs, action):
        """
        Modified step function used for value iteration / policy iteration:
        Execute a step with the agent in the Gridworld environment. Perform an action for a given state
        and obtain an observation, a reward and the information (done) whether the agent ended up in a terminal state
        :param obs: Scalar value of the state
        :param action: Scalar value of the action (UP/RIGHT/DOWN/LEFT) according to config file
        :return: (state, reward, done flag)
        """
        # use this function only for value iteration / policy iteration
        self.state = self._obs_to_state(obs)
        return self.step(action)


    def reset(self):
        """
        Resets the environment, must be called at the beginning of every episode
        """
        m, n = self.shape()

        # find start state
        for x in range(m):
            for y in range(n):
                if self.grid[x][y] == 'S':
                    self.state = (x, y)
                    return self._state_to_obs(self.state)

        raise ValueError("No start state found")


    def _calc_next_state(self, action):
        m, n = self.shape()
        x, y = self.state

        # hole/goal -> stuck
        if self.grid[x][y] == 'O' or self.grid[x][y] == 'G':
            return x, y

        # calculate movement direction
        if action == G_LEFT:
            x_n, y_n = x, y - 1
        elif action == G_UP:
            x_n, y_n = x - 1, y
        elif action == G_RIGHT:
            x_n, y_n = x, y + 1
        elif action == G_DOWN:
            x_n, y_n = x + 1, y
        else:
            raise ValueError('Unknown action: ' + str(action))

        # movement limited by map boundaries
        x_n = min(max(x_n, 0), m - 1)
        y_n = min(max(y_n, 0), n - 1)

        # movement limited by wall boundaries
        if self.grid[x_n][y_n] == '#':
            x_n, y_n = x, y

        # fly to other airport
        if self.grid[x_n][y_n] == 'A':
            for a in range(len(self.grid)):
                for b in range(len(self.grid[a])):
                    if self.grid[a][b] == 'A' and a != x_n and b != y_n:
                        x_a = a
                        y_a = b
            x_n = x_a
            y_n = y_a

        return x_n, y_n


    # calculate reward flag
    def _calc_reward(self):
        x, y = self.state
        if self.grid[x][y] == 'G':
            return self.g_reward
        elif self.grid[x][y] == 'O':
            return self.o_reward
        else:
            return self.step_cost


    # calculate done flag
    def _calc_done(self):
        x, y = self.state
        if self.grid[x][y] in ('O', 'G'):
            return True
        else:
            return False


    # convert from observation (int) to internal state representation (x,y)
    def _obs_to_state(self, obs):
        n = len(self.grid[0])
        x = obs // n
        y = obs % n
        return x, y


    # convert from internal state representation (x,y) to observation (int)
    def _state_to_obs(self, state):
        n = len(self.grid[0])
        x, y = state
        obs = x * n + y
        return obs


class Test(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', 'A'],
                     [' ', ' ', ' '],
                     ['A', ' ', 'G']]
        GridworldEnv.__init__(self)


class EmptyWorld33(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', ' '],
                     [' ', ' ', ' '],
                     [' ', ' ', 'G']]
        GridworldEnv.__init__(self)


class EmptyWorld55(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', 'G']]
        GridworldEnv.__init__(self)


class MountainWorld(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', '#', ' ', ' '],
                     [' ', ' ', '#', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', '#', ' ', ' '],
                     [' ', ' ', '#', ' ', 'G']]
        GridworldEnv.__init__(self)


class WaterWorld(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', 'O', ' ', ' '],
                     [' ', ' ', 'O', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', 'O', ' ', ' '],
                     [' ', ' ', 'O', ' ', 'G']]
        GridworldEnv.__init__(self)


class Cliff(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     ['S', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'G']]
        GridworldEnv.__init__(self)


class ExerciseWorld(GridworldEnv):
    def __init__(self):
        self.step_cost = -1
        self.g_reward = -1
        self.o_reward = -100
        self.grid = [['S', ' ', ' ', ' ', 'A', ' ', 'O'],
                     [' ', ' ', '#', ' ', ' ', ' ', ' '],
                     ['A', ' ', ' ', ' ', '#', ' ', 'G']]
        GridworldEnv.__init__(self)


class MazeWater(GridworldEnv):
    def __init__(self):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     ['S', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' ', 'O', 'G']]
        GridworldEnv.__init__(self)


class MazeWater2(GridworldEnv):
    def __init__(self):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ' '],
                     [' ', 'O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
                     [' ', 'O', ' ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                     [' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' ', ' ', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', 'O', 'O', 'O', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', ' ', ' ', ' ', 'O', ' ', 'O', ' '],
                     [' ', 'O', ' ', 'O', 'O', 'O', 'O', 'O', ' ', 'O', ' '],
                     ['S', 'O', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'O', 'G']]
        GridworldEnv.__init__(self)


class MazeWall(GridworldEnv):
    def __init__(self):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' ', ' ', ' ', '#', ' ', ' ', ' ', '#', ' ', ' ', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     [' ', '#', ' ', '#', ' ', '#', ' ', '#', ' ', '#', ' '],
                     ['S', '#', ' ', ' ', ' ', '#', ' ', ' ', ' ', '#', 'G']]
        GridworldEnv.__init__(self)


class EmptyWorldNN(GridworldEnv):
    # create an empty quadratic grid with the start in one corner and the goal in the opposite corner
    def __init__(self, size=10):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100
        self.grid = [[' '] * size for _ in range(size)]
        self.grid[0][0] = 'S'
        self.grid[size-1][size-1] = 'G'
        GridworldEnv.__init__(self)


class Random(GridworldEnv):
    # create a random quadratic environment with a given percentage of all cells covered by water/mountains
    def __init__(self, size=10, water=0.1, mountain=0.1, valid_path_guaranteed=True):
        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100

        # create new environments until one with a valid start-goal path is found
        if valid_path_guaranteed:
            valid_grid = False
            while not valid_grid:
                self._create_grid(size, water, mountain)
                valid_grid = self._check_valid_grid()
        else:
            self._create_grid(size, water, mountain)

        GridworldEnv.__init__(self)


    # create the environment
    def _create_grid(self, size, water, mountain):
        self.grid = [[' ']*size for _ in range(size)]

        # set water cells
        for i in range(int(size**2*water)):
            self.grid[random.randint(0,size-1)][random.randint(0,size-1)] = 'O'

        # set mountain cells
        for i in range(int(size**2*mountain)):
            self.grid[random.randint(0,size-1)][random.randint(0,size-1)] = '#'

        # define start
        while True:
            r_start, c_start = random.randint(0,size-1), random.randint(0,size-1)
            # prevent start from being in the center of the map
            if abs(r_start-size/2) < size/3 and abs(c_start-size/2) < size/3:
                continue
            break

        # define goal
        while True:
            r_end, c_end = random.randint(0,size-1), random.randint(0,size-1)
            # prevent goal from being in the center of the map
            if abs(r_end-size/2) < size/3 and abs(r_end-size/2) < size/3:
                continue
            # prevent goal from being too close to start
            if abs(r_end-r_start) < size/2 and abs(c_end-c_start) < size/2:
                continue
            if r_end != r_start or c_end != c_start:
                break

        # clear area around start/goal from obstacles
        clear_pixel = 1
        for r, c in ((r_start, c_start), (r_end, c_end)):
            for i in range(max(0, r-clear_pixel), min(size, r+clear_pixel+1)):
                for k in range(max(0, c-clear_pixel), min(size, c+clear_pixel+1)):
                    self.grid[i][k] = ' '

        # set start/goal
        self.grid[r_start][c_start] = 'S'
        self.grid[r_end][c_end] = 'G'


    # check if a path from start to goal can be found
    def _check_valid_grid(self):
        size = len(self.grid)
        visited = [[False]*size for _ in range(size)]

        def dfs(row, col):
            if row < 0 or col < 0 or row >= size or col >= size or \
                    self.grid[row][col] in ['#', 'O'] or \
                    visited[row][col] is True:
                return False

            visited[row][col] = True

            if self.grid[row][col] == 'G' or \
                    dfs(row + 1, col) or \
                    dfs(row - 1, col) or \
                    dfs(row, col + 1) or \
                    dfs(row, col - 1):
                return True

            return False

        for row in range(size):
            for col in range(size):
                if self.grid[row][col] == 'S':
                    return dfs(row,col)

        return ValueError("No start found")


class Random_Switch(Random):
    def __init__(self, size=12, water=0.4, mountain=0.0, valid_path_guaranteed=False):
        self.step_num = 0
        super().__init__(size, water, mountain, valid_path_guaranteed)

    def reset(self):
        self.step_num = 0
        super().reset()

    def step(self, action):
        self.step_num += 1

        if self.step_num % 5 != 0:
            return super().step(action)
        else:
            m, n = self.shape()
            x, y = self.state

            # calculate movement direction
            if action == G_LEFT:
                x_n, y_n = x, y - 1
            elif action == G_UP:
                x_n, y_n = x - 1, y
            elif action == G_RIGHT:
                x_n, y_n = x, y + 1
            elif action == G_DOWN:
                x_n, y_n = x + 1, y
            else:
                raise ValueError('Unknown action: ' + str(action))

            # movement limited by map boundaries
            x_n = min(max(x_n, 0), m - 1)
            y_n = min(max(y_n, 0), n - 1)

            self.state = (x_n, y_n)

            if (x_n + y_n) % 2 == 0:
                reward = -100
                done = True
            else:
                reward = 0
                done = False

            obs = self._state_to_obs(self.state)

            return obs, reward, done


class SeaShanty(GridworldEnv):
    """
    A dynamic Gridworld environment with wave patterns
    """

    def __init__(self, size=12, period=4):
        """
        Initialize SeaShanty Gridworld

        Params:
            size: Size of the grid (size x size)
            period: Period of wave pattern changes
        """

        self.step_cost = 0
        self.g_reward = 100
        self.o_reward = -100

        self.size = size
        self.period = period

        self._grid = np.zeros((self.size, self.size), dtype=float)
        self.grid = np.full((self.size, self.size), ' ', dtype=str)

        self._step_num = 0
        self._rotation_x = np.random.choice([-1, 1])
        self._rotation_y = np.random.choice([-1, 1])

        self._start = (0, 0)
        self._goal = (self.size - 1, self.size - 1)
        self._place_start_goal()

        self.ix, self.iy = np.indices((self.size, self.size))

        self._draw_waves()
        self._draw_start_goal()
        self._grid2grid()
        GridworldEnv.__init__(self)


    def reset(self):
        """
        Resets the environment, must be called at the beginning of every episode
        """
        self._step_num = 0
        return super().reset()


    def _distance(self, pos1, pos2):
        """
        Calculate Manhattan distance between two positions

        Params:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)

        Returns:
            Manhattan distance between pos1 and pos2
        """

        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


    def _place_start_goal(self):
        """
        Place start and goal positions randomly, ensuring they are sufficiently apart
        """

        i = 0
        while True:
            i += 1
            start_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            goal_pos = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if self._distance(start_pos, goal_pos) >= self.size:
                self._start = start_pos
                self._goal = goal_pos
                return
            if i > 100:
                self._start = (0, 0)
                self._goal = (self.size - 1, self.size - 1)
                return


    def _draw_start_goal(self):
        """
        Draw fixed, start and goal positions on the grid
        """

        self._grid[self._start] = 10
        self._grid[self._goal] = -10


    def _draw_waves(self):
        """
        Draw dynamic wave patterns on the grid
        """
        self._grid = np.sin(self.ix + (self._rotation_x * (self._step_num / self.period))) + \
                     np.cos(self.iy + (self._rotation_y * (self._step_num / self.period)))
        self._step_num += 1


    def _grid2grid(self):
        """
        Convert numerical grid to character grid representation
        """

        self.grid.fill(' ')

        water = np.abs(self._grid) >= 1.125
        self.grid[water] = 'O'

        self.grid[self._grid == 10] = 'S'
        self.grid[self._grid == -10] = 'G'


    def _render_grid(self, obs=None):
        """
        Convert numerical grid to RGB image for visualization

        Returns:
            img: Numpy array representing the RGB image of the grid
        """

        img = np.zeros((*self._grid.shape, 3), dtype=np.uint8)

        water = np.abs(self._grid) >= 1.2

        img[water] = [0, 0, 255]  # water
        img[self._grid == 10] = [0, 255, 0]  # start
        img[self._grid == -10] = [255, 0, 0]  # goal

        if obs is not None:
            state = self._obs_to_state(obs)
            img[state] = [200, 200, 200]  # agent pos

        return img


    def step(self, action):
        """
        Execute a step with the agent in the Gridworld environment

        Params:
            action: Scalar value of the action (UP/RIGHT/DOWN/LEFT) according to config file

        Returns:
            obs: New observation after taking the action
            reward: Reward obtained after taking the action
            done: Boolean flag indicating if the episode has ended
        """
        # Redraw with updated cross pattern
        self._draw_waves()
        self._draw_start_goal()
        self._grid2grid()

        return super().step(action)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    reps = 10
    steps = 100

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))

    env = SeaShanty(size=6, period=6)

    for _ in range(reps):
        obs = env.reset()

        for _ in range(steps):
            img = env._render_grid(obs)

            ax.clear()
            ax.imshow(img)
            ax.set_title("SeaShanty Environment")
            ax.axis('off')
            plt.pause(0.5)

            # for demonstration purpose only
            action = np.random.randint(0, env.num_actions())
            obs, reward, done = env.step(action)

    plt.ioff()
    plt.show()
