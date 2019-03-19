import tools
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle

# TODO: Find a way to make this generate mazes faster
class BoxGame(object):
    """
    A simple box game

    Grid key:
        0 ~ empty space
        +1 ~ agent
        -1 ~ blocked
        +2 ~ exit
        -2 ~ pits i.e. false exits

    """

    def __init__(self,
                 length,
                 block_rate=.2,
                 num_exits=1,
                 num_empty_targets=0,
                 num_pits=0,
                 watch_construction=False,
                 max_tree_steps=1000,
                 bias=None,
                 pad_size=5,
                 instant_death_p=None,
                 timeout_steps=None):

        self.size = (length, length)
        self.grid = torch.zeros(self.size)
        self.start_grid = torch.zeros(self.size)

        self.exit_flag = False
        self.pit_flag = False
        self.exit_reward = .2
        self.pit_reward = -1.
        self.step_reward = -.1
        self.crash_reward = -.1
        self.instant_death_reward = self.crash_reward
        self.timeout_reward = -.1

        self.agent_point = np.array([0,0])
        self.start_point = np.array([0,0])
        self.exit_points = []
        self.original_exit_points = []
        self.pit_points = []
        self.original_pit_points = []

        self.watch_construction = watch_construction
        self.max_tree_steps = max_tree_steps
        self.bias = bias
        self.pad_size = pad_size

        self.block_rate = block_rate
        self.num_exits = num_exits
        self.num_pits = num_pits
        self.num_empty_targets = num_empty_targets

        self.instant_death_p = instant_death_p

        self.timeout_steps = timeout_steps
        self.counter = 0

        self.key = {"open": 0.,
                    "agent": 0.,
                    "block": -1.,
                    "exit": 2.,
                    "pit": -2.}

        # How far apart the start and exits are
        self.exit_distance = length/2
        self.reserved = []

        self.fill_grid()

        self.pad_grid()

    def reset_grid(self):

        self.exit_flag = False
        self.pit_flag = False
        self.counter = 0

        self.agent_point = self.start_point.copy()

        for i in range(self.num_exits):
            self.exit_points[i] = self.original_exit_points[i].copy()

        for i in range(self.num_pits):
            self.pit_points[i] = self.original_pit_points[i].copy()

        self.grid = self.start_grid.clone()

        self.pad_grid()

    def agent_view(self, agent_range):

        delta = np.array([agent_range,agent_range])
        low = self.agent_point - delta
        high = self.agent_point + delta

        out = self.grid[low[0]:(high[0]+1), low[1]:(high[1]+1)]
        return out.contiguous()

    def move_agent(self, step):
        """
        Moves the agent by the specified array `step`.
        :param array step:
        `step` is a 1d array with length 2.  Both entries of the array must
        be type int.

        :returns tuple (float reward, bool terminal):
        `terminal` is True if the agent has stepped onto an exit or pit,
         False otherwise.
        """
        if self.instant_death_p is not None:
            if np.random.random() < self.instant_death_p:
                reward = self.instant_death_reward
                terminal = True
                return reward, terminal


        new_point = self.agent_point + step

        blocked = self.grid[tuple(new_point)] == self.key["block"]
        self.exit_flag = self.grid[tuple(new_point)] == self.key["exit"]
        self.pit_flag = self.grid[tuple(new_point)] == self.key["pit"]

        if self.timeout_steps and not (self.exit_flag or self.pit_flag):
            if self.counter >= self.timeout_steps:
                reward = self.timeout_reward
                terminal = True
                return reward, terminal

            self.counter += 1

        # First we check if the agent is trying to move onto a blocked point.
        # If so, reject the step and count it as a normal step for the reward.
        if blocked:
            reward = self.crash_reward
            terminal = False

            return reward, terminal

        else:
            # move agent
            self.grid[tuple(self.agent_point)] = self.key["open"]
            self.agent_point += step
            self.grid[tuple(self.agent_point)] = self.key["agent"]

        if self.exit_flag:
            reward = self.exit_reward
            terminal = True

        elif self.pit_flag:
            reward = self.pit_reward
            terminal = True

        else:
            reward = self.step_reward
            terminal = False

        return reward, terminal

    def fill_grid(self):
        """
        Initializes the  grid with obstacles.


        Grid key:
            0 ~ empty space
            +1 ~ agent
            -1 ~ blocked
            +2 ~ exit
            -2 ~ pits i.e. false exits
        """
        self.start_point = self.rand_point()
        self.agent_point = self.start_point.copy()
        self.grid[tuple(self.agent_point)] = self.key["agent"]


        # self.reserved = []

        for i in range(self.num_exits):

            exit_point = self.rand_point(unused_only=True)

            while abs(exit_point - self.agent_point).sum() < self.exit_distance:
                exit_point = self.rand_point(unused_only=True)

            self.grid[tuple(exit_point)] = self.key["exit"]
            self.exit_points.append(exit_point)
            self.original_exit_points.append(exit_point)

            reserve_tree = self.generate_tree(exit_point)
            while reserve_tree.fail:
                reserve_tree = self.generate_tree(exit_point)

            self.reserved += reserve_tree.path

        for i in range(self.num_empty_targets):

            target_point = self.rand_point(unused_only=True)

            reserve_tree = self.generate_tree(target_point)
            while reserve_tree.fail:
                reserve_tree = self.generate_tree(target_point)

            self.reserved += reserve_tree.path

        for i in range(self.num_pits):

            pit_point = self.rand_point(unused_only=True)

            while abs(pit_point - self.agent_point).sum() < self.exit_distance:
                pit_point = self.rand_point(unused_only=True)

            self.grid[tuple(pit_point)] = self.key["pit"]
            self.pit_points.append(pit_point)
            self.original_pit_points.append(pit_point)

            reserve_tree = self.generate_tree(pit_point)
            while reserve_tree.fail:
                reserve_tree = self.generate_tree(pit_point)



            self.reserved += reserve_tree.path

        num_remaining = self.size[0]**2 - len(self.reserved)
        num_blocks = int(self.block_rate * num_remaining)

        for i in range(num_blocks):
            block = self.rand_point(unused_only=True)
            self.grid[tuple(block)] = self.key["block"]

        # Save the starting grid in case we need to reset it
        self.start_grid = self.grid.clone()

    def pad_grid(self):

        pad = self.pad_size

        pad_layer = torch.nn.ConstantPad2d(pad, self.key["block"])
        self.grid = pad_layer(self.grid)

        # Now correct the agent, exit, and pit points.
        shift = np.array([pad, pad])

        self.agent_point += shift
        for i in range(self.num_exits):
            self.exit_points[i] += shift
        for i in range(self.num_pits):
            self.pit_points[i] += shift

    def generate_tree(self, end):

        tree = ReservingTree(self.size[0],
                             self.agent_point,
                             end,
                             watch=self.watch_construction,
                             max_steps=self.max_tree_steps,
                             bias=self.bias)

        return tree

    def rand_point(self, unused_only=False):
        """
        Returns a random point in the grid.


        """
        length = self.size[0]
        x = np.random.randint(length, size=(2,))

        if unused_only:
            free = tools.almost_same(self.grid[tuple(x.tolist())], 0)
            reserved = False
            for y in self.reserved:
                reserved = reserved or np.array_equal(x, y)

            while (not free) or reserved:
                x = np.random.randint(length, size=(2,))

                free = tools.almost_same(self.grid[tuple(x.tolist())], 0)
                reserved = False
                for y in self.reserved:
                    reserved = reserved or np.array_equal(x, y)

        return x

    def show_grid(self, show_agent=True):
        if show_agent:
            grid = self.grid.clone()
            grid[tuple(self.agent_point)] = 1.
        else:
            grid = self.grid

        plt.matshow(grid.numpy(),vmin=-2., vmax=2.)
        plt.show(block=False)


class ReservingTree(object):

    def __init__(self, grid_length, start, finish,
                 watch=False, max_steps=1000, bias=None):

        self.fail = False
        self.grid_length = grid_length
        self.start = start
        self.finish = finish
        self.bias = bias

        self.grid = None

        if watch:
            self.grid = np.zeros((grid_length, grid_length))
            self.grid[tuple(start)] = 1.
            self.grid[tuple(finish)] = 2.

        # TODO: Consider making this a tuple
        self.steps = [np.array([1, 0]),
                      np.array([0, 1]),
                      np.array([-1, 0]),
                      np.array([0, -1])]

        self.path = [start]
        self.forward_list = []
        self.short_list = []

        candidates = []
        for i in range(4):

            forward_point = self.start + self.steps[i]
            if self.on_grid(forward_point):
                candidates.append(forward_point)
                #short_cuts.append(forward_point)

        self.short_list.append(candidates.copy())

        action_index = np.random.randint(len(candidates))
        point = candidates.pop(action_index)
        self.path.append(point)

        if watch:
            self.grid[tuple(point)] = 1.5

        self.forward_list.append(candidates)
        t = 2
        step_count = 0
        terminate = False
        while (not terminate) and (not self.fail): # TODO: remove
            step_count += 1
            if step_count > max_steps:
                self.fail = True
                print("ReservingTree failed to generate a path in"
                      " step limit. Consider changing optional"
                      " argument in ReservingTree.__init__ max_steps")

            if watch:
                self.show_path()

            if len(self.path) > t:
                candidates = self.forward_list[-1]
                del self.forward_list[-1]

                if len(self.short_list) > t:
                    del self.short_list[-1]

                if watch:
                    self.grid[tuple(self.path[t])] = 0.
                    self.show_path()

                del self.path[t]

            else:
                candidates = []

                point = self.path[-1]
                previous_point = self.path[-2]
                for i in range(4):

                    forward_point = point + self.steps[i]
                    if np.array_equal(forward_point, finish):
                        terminate = True

                    back_flag = np.array_equal(forward_point, previous_point)
                    on_flag = self.on_grid(forward_point)
                    shortcut_flag = self.shortcut(forward_point)

                    if on_flag and (not back_flag) and (not shortcut_flag):
                        candidates.append(forward_point)

            if terminate:
                self.path.append(finish)
                self.forward_list.append("terminal")
                t = t + 1

            elif candidates:

                if len(self.short_list) == len(self.forward_list):
                    self.short_list.append(candidates.copy())

                action_index = np.random.randint(len(candidates))
                action_index = self.pick_action(candidates)

                next_point = candidates.pop(action_index)

                self.path.append(next_point)

                self.forward_list.append(candidates)
                t = t + 1

                # TODO: clean
                if watch:
                    self.grid[tuple(next_point)] = 1.5

            else:
                t = t - 1

    def pick_action(self, candidates):

        if self.bias is None:
            return np.random.randint(len(candidates))

        distances = []
        min_dist_indices = []
        for i in range(len(candidates)):
            dist = abs(candidates[i] - self.finish).sum()
            distances.append(dist)

        min_distance = np.min(distances)

        for i in range(len(distances)):
            if tools.almost_same(distances[i], min_distance):
                min_dist_indices.append(i)

        uniform_choice = np.random.randint(len(distances))
        directed_choice = np.random.choice(min_dist_indices)

        rand = np.random.random()

        if rand < self.bias:
            choice = directed_choice

        else:
            choice = uniform_choice

        return choice




    def shortcut(self, point):

        for t in range(len(self.short_list)):

            for k in range(len(self.short_list[t])):
                if np.array_equal(point, self.short_list[t][k]):
                    return True

        return False

    def on_grid(self, point):

        x_on = point[0] in range(self.grid_length)
        y_on = point[1] in range(self.grid_length)

        return x_on and y_on

    def show_path(self):

        plt.matshow(self.grid,cmap="Greys_r")
        plt.show(block=False)
        plt.pause(.001)
        plt.close()


def generate_mazes(num_gen, filename, size, pad_size, block_rate, num_exits,
                   timeout_steps, num_empty_targets=0, instant_death_p=None,
                   generator_bias=.9, max_gen_steps=150):
    mazes = []
    for i in range(num_gen):
        mazes.append(
            BoxGame(length=size,
                    block_rate=block_rate,
                    num_exits=num_exits,
                    num_empty_targets=num_empty_targets,
                    num_pits=0,
                    max_tree_steps=max_gen_steps,
                    bias=generator_bias,
                    pad_size=pad_size,
                    instant_death_p=instant_death_p,
                    timeout_steps=timeout_steps
                    )
        )
        if i % 200 == 0:
            print("maze generator progress: ", int(100 * i/num_gen), "%")
    print("maze generator progress: 100%")

    pik_name = filename
    with open(pik_name, "wb") as f:
        pickle.dump(mazes, f)


if __name__=="__main__":


    print("")
    generate_mazes(10000, "maze_3_17.dat",
                   size=15,
                   pad_size=5,
                   block_rate=1.,
                   num_exits=1,
                   num_empty_targets=1,
                   timeout_steps=750,
                   generator_bias=.4,
                   max_gen_steps=400
                   )



