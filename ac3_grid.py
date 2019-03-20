from collections import namedtuple
from multiprocessing import Process, Queue, current_process, set_start_method
import gridworld
import random
import tools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import networks
import pickle
from gridworld import BoxGame


class Agent(object):
    def __init__(self, field_ch, num_temp_frames, ag_range, memory_coef,
                 ac_state_dict=None):
        # TODO: temp removal of field (change back the type of network)
        self.field_ch = field_ch
        self.num_temp_frames = num_temp_frames
        self.range = ag_range
        assert self.range == 5, "Currently, range must be 5."
        self.memory_coef = memory_coef
        self.preprocessor = networks.SensoryNetFixed()

        self.ac_net = networks.ActorCriticNet1F(self.field_ch,
                                                  self.num_temp_frames)
        if ac_state_dict:
            self.ac_net.load_state_dict(ac_state_dict)

        self.environment = None
        width = 2 * self.range + 1
        self.view = torch.zeros((width, width))
        self.frames = None
        self.field_size = (self.field_ch, 28, 28)
        self.field = None

        self.steps = [
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
            np.array([1, 0])
        ]

    def episode_run(self, env=None):
        if env:
            self.set_environment(env)
        terminal = False
        while not terminal:
            (move, write) = self.pick_action()
            _, terminal = self.show_move(move.item())
            self.update_field(write)

    def set_environment(self, env):
        if self.environment:
            self.environment.reset_grid()
        self.environment = env
        self.initialize_frames()
        self.initialize_field()

    def initialize_frames(self):
        view = self.get_view()
        view_stack = torch.stack([view for _ in range(self.num_temp_frames)])
        self.frames = self.preprocessor(view_stack)

    def initialize_field(self):
        self.field = torch.zeros(self.field_size)

    def get_view(self, noise_scale=.01):
        # if self.view_loaded:
        #    return self.view
        self.view = self.environment.agent_view(self.range)
        self.view += noise_scale * torch.randn(self.view.size())
        return self.view

    def move(self, move_index):
        """
        :param int move_index: an index for self.steps
        :return: (float reward, bool terminal):
        reward: the reward the environment returns after the action
        terminal: True if the new state is terminal, False otherwise
        """
        # d
        #
        step = self.steps[move_index]
        (reward, terminal) = self.environment.move_agent(step)
        view = self.get_view()
        frame = self.preprocessor(view.unsqueeze(0))
        self.frames = torch.cat((self.frames[1:], frame))
        return reward, terminal

    def update_field(self, change):
        self.field = self.memory_coef * self.field + change
        self.field.clamp_(-10., 10.)

    def pick_action(self, log_probs: bool):
        """
        """
        move_log_probs = self.ac_net(self.frames, self.field, "move")
        move_probs = torch.exp(move_log_probs)
        move_dist = torch.distributions.categorical.Categorical(move_probs)
        move_action = move_dist.sample()
        write_dist = self.ac_net(self.frames, self.field, "write")
        write = write_dist.sample()

        if log_probs:
            move_log_prob = move_log_probs[move_action]
            write_log_prob = write_dist.log_prob(write).sum()
            return (move_action, write), (move_log_prob, write_log_prob)

        return move_action, write

    def show_view(self):
        view = self.view.clone()
        view[self.range, self.range] = 1.

        plt.matshow(view.numpy(), cmap="Greys", vmin=-2., vmax=2.)
        plt.show(block=False)
        plt.pause(.02)

    def show_move(self, direction, show_full_grid=False, show_field=False):

        if isinstance(direction, str):
            key_map = {
                "W": 1,
                "A": 2,
                "S": 3,
                "D": 0,
                "w": 1,
                "a": 2,
                "s": 3,
                "d": 0
            }
            direction = key_map[direction]

        plt.close()
        (reward, terminal) = self.move(direction)
        # self.get_view()

        if show_full_grid:
            view = self.view.clone()
            grid = self.environment.grid.clone()

            view[self.range, self.range] = 1.
            grid[tuple(self.environment.agent_point)] = 1.
            tools.show_images(
                [view.numpy(), grid.numpy()],
                titles=["Agent perspective", "Maze map"]
            )

        elif show_field:
            view = self.view.clone()
            view[self.range, self.range] = 1.
            fields = [.2 * self.field[i].clone().numpy()
                      for i in range(self.field_ch)]

            tools.show_images([view.numpy()] + fields)

        else:
            self.show_view()

        return reward, terminal

    def show_internal_field(self):
        plt.close()
        plt.imshow(self.internal_field[0].numpy(), vmin=-10., vmax=10.)
        plt.show()
        plt.pause(.001)

    def play(self, show_grid=False, discount=.95):

        self.show_view(get_view=True)

        key_map = {
            "W": 1,
            "A": 2,
            "S": 3,
            "D": 0,
            "w": 1,
            "a": 2,
            "s": 3,
            "d": 0
        }

        t = 0
        total_return = 0
        terminal = False

        while not terminal:

            action = input("Use WASD to move (you have to push enter--sorry).")
            try:
                (reward, terminal) = self.show_move(key_map[action],
                                                    show_full_grid=show_grid)
                total_return += discount ** t * reward
                t += 1

            except KeyError:
                print("input error")

        print("END! Final return is", total_return)


Params = namedtuple("Params",
                    (
                        "discount", "td_steps", "range", "field_ch",
                        "temp_frames",
                        "memory_coef", "global_t_max", "entropy_coef",
                        "recorder_batch_size"))

SharedQueues = namedtuple("SharedQueues",
                          ("maze_queue", "param_queue", "glb_t_queue"))

LevelRanges = namedtuple("LevelRanges",
                         ("num_samples", "range_list"))


class AC3Process(Agent):
    def __init__(self,
                 queues: SharedQueues,
                 params: Params,
                 level_ranges: LevelRanges = None):
        Agent.__init__(self, params.field_ch, params.temp_frames, params.range, params.memory_coef, None)

        if queues.maze_queue:
            self.mode = "maze_queue"
        elif level_ranges:
            self.mode = "level_ranges"
        else:
            raise RuntimeError("Unable to determine maze gathering protocol.")

        self.env_list = []
        self.level_ranges = level_ranges
        self.track = True

        self.discount = params.discount
        self.td_steps = params.td_steps

        self.glb_t_max = params.global_t_max
        self.entropy_coef = params.entropy_coef  # TODO implement entropy loss

        self.maze_queue = queues.maze_queue
        self.param_queue = queues.param_queue
        self.glb_t_queue = queues.glb_t_queue

        critic_params = [
            {'params': self.ac_net.field_conv.parameters()},
            {'params': self.ac_net.temporal_conv.parameters()},
            {'params': self.ac_net.processing.parameters()},
            {'params': self.ac_net.critic.parameters()}
        ]
        move_params = [
            {'params': self.ac_net.field_conv.parameters()},
            {'params': self.ac_net.temporal_conv.parameters()},
            {'params': self.ac_net.processing.parameters()},
            {'params': self.ac_net.log_pol_move.parameters()}
        ]
        write_params = [
            {'params': self.ac_net.field_conv.parameters()},
            {'params': self.ac_net.temporal_conv.parameters()},
            {'params': self.ac_net.processing.parameters()},
            {'params': self.ac_net.mean_generator.parameters()},
            {'params': self.ac_net.sd_generator.parameters()}
        ]
        discriminator_params = [
            {'params': self.ac_net.field_conv.parameters()},
            {'params': self.ac_net.discriminator_end.parameters()}
        ]
        critic_opt = optim.RMSprop(critic_params, lr=2e-5)
        move_opt = optim.RMSprop(move_params, lr=1e-5) # TODO: focus on memory
        write_opt = optim.RMSprop(write_params, lr=1e-5)
        discriminator_opt = optim.RMSprop(discriminator_params, lr=1e-6)
        #self.optimizers = [critic_opt, move_opt, write_opt, discriminator_opt]
        self.optimizers = [discriminator_opt]  # TODO revert
        self.mem_opt_threshold = .1
        self.asynchronous_update(optimize=False)
        self.glb_t = 0

        self.recorder = Recorder(params.recorder_batch_size)

        # Load the first environment and initialize self.frames and self.field.
        self.next_environment()

    def run(self):

        t = 1
        # move_action_list = torch.zeros(self.td_steps, dtype=torch.int)
        # write_action_list = torch.zeros(
        #    torch.Size(self.td_steps) + self.field_size)
        if self.track:
            crash_rew = 0.
            step_rew = 0.
            exit_rew = 1.
            crashes = 0
            steps = 0
            exits = 0

        reward_list = torch.zeros(self.td_steps)
        state_list = [None for _ in range(self.td_steps + 1)]
        move_log_prob_list = [None for _ in range(self.td_steps)]
        write_log_prob_list = [None for _ in range(self.td_steps)]

        terminal = False
        show = False
        show_interval = 50000
        show_thresh = show_interval
        while self.glb_t < self.glb_t_max and self.environment:
            global_t_increment = 0
            t_start = t
            state_list[0] = (self.frames, self.field)

            while t - t_start < self.td_steps and not terminal:
                self.record()
                (mv_act, write), (mv_lp, write_lp) = self.pick_action(True)
                if show:
                    reward, terminal = self.show_move(mv_act.item(), show_field=True)
                else:
                    reward, terminal = self.move(mv_act.item())
                self.update_field(write)

                if self.track:
                    if reward == crash_rew:
                        crashes += 1
                    if reward == step_rew:
                        steps += 1
                    if reward == exit_rew:
                        exits += 1

                reward_list[t - t_start] = reward
                move_log_prob_list[t - t_start] = mv_lp
                write_log_prob_list[t - t_start] = write_lp
                t += 1
                global_t_increment += 1
                state_list[t - t_start] = (self.frames, self.field)

            if terminal:
                ret = 0.
                self.recorder.finish_episode()
                reward_list[t - 1 - t_start] += self.memory_test()
                self.next_environment()
                terminal = False

                show = False
                if t > show_thresh:
                    show = True
                    show_thresh += show_interval
                if self.track:
                    print("{:6d} {:6d} {:6d} {:6d}"
                          .format(crashes, steps, exits, self.glb_t))
                    crashes = 0
                    steps = 0
                    exits = 0
            else:
                ret = self.ac_net(
                    self.frames, self.field, mode="critic").detach()

            for i in reversed(range(t_start, t)):
                ret = reward_list[i - t_start] + self.discount * ret
                (frames, field) = state_list[i - t_start]
                # move_action = move_action_list[i - t_start]
                # write_action = write_action_list[i - t_start]

                move_lp = move_log_prob_list[i - t_start]
                write_lp = write_log_prob_list[i - t_start]

                value = self.ac_net(frames, field, "critic")

                move_loss = -move_lp * (ret - value.detach())
                move_loss.backward()

                write_loss = -write_lp * (ret - value.detach())
                write_loss.backward()

                value_loss = (ret - value) ** 2
                value_loss.backward()
                if random.random() < .001:
                    print(torch.exp(move_log_prob_list[i - t_start]))
                    print("val:  ", value.item())
                    print("ret:  ", ret)
                    print("move loss: ", move_loss)

            self.change_global_time(global_t_increment)
            self.asynchronous_update(optimize=True)

    def asynchronous_update(self, optimize: bool):
        """
        When optimize is False, this method simply loads the latest
        shared state_dict to self.ac_net (the network for this process).

        If optimize is True, this method loads the latest shared state_dict,
        changes it according to the accumulated gradients since the last
        update, and then stores the updated state_dict onto the shared
        parameter queue.

        :param optimize: bool
        """
        params = self.param_queue.get()
        self.ac_net.load_state_dict(params)

        if optimize:
            for opt in self.optimizers:
                opt.step()
                opt.zero_grad()
            params = self.ac_net.state_dict()

        self.param_queue.put(params)

    def change_global_time(self, change):
        t = self.glb_t_queue.get()
        t = t + change

        self.glb_t = t
        self.glb_t_queue.put(t)

    def record(self):
        self.recorder.save(self.frames[-1], self.field)

    def memory_test(self, num_cycles=2):

        extracted = self.recorder.extract()
        if not extracted:
            return 0.

        frame_back, frame_fwd, field_fwd = extracted
        print(field_fwd[0][0])

        confidence = self.ac_net(frame_fwd, field_fwd, "test_memory")
        rejection = 1. - self.ac_net(frame_back, field_fwd, "test_memory")

        reward = (confidence[-1] + rejection[-1]) / 2.
        reward = reward.item()

        loss = -torch.log(confidence) - torch.log(rejection)
        loss = loss.mean()

        if loss > self.mem_opt_threshold:
            loss.backward()

            for _ in range(num_cycles - 1):
                extracted = self.recorder.extract()
                frame_back, frame_fwd, field_fwd = extracted

                confidence = self.ac_net(frame_fwd, field_fwd, "test_memory")
                rejection = 1. - self.ac_net(frame_back, field_fwd, "test_memory")
                loss = -torch.log(confidence) - torch.log(rejection)
                loss = loss.mean()
                loss.backward()

        print(loss.item())
        return reward

    def next_environment(self):
        """
        TODO new doc
        """
        if self.environment:
            self.environment.reset_grid()
        if self.mode == "maze_queue":
            if self.maze_queue.empty():
                self.environment = None
            else:
                self.environment = self.maze_queue.get()
        elif self.mode == "level_ranges":
            try:
                self.environment = self.env_list.pop(0)
            except IndexError:
                # load_mazes returns None if the level range list is finished.
                self.environment = self.load_mazes()
        else:
            raise RuntimeError("self.mode not recognized")

        if self.environment:
            self.initialize_frames()
            self.initialize_field()

    def load_mazes(self):
        try:
            (min_lev, max_lev) = self.level_ranges.range_list.pop(0)
        except IndexError:
            return None
        for lev in range(min_lev, max_lev + 1):
            print("\n\n***********************\n\nloading level ", lev, "\n\n")
            filename = "level_" + str(lev) + ".dat"
            with open(filename, "rb") as f:
                maze_group = pickle.load(f)
        random.shuffle(maze_group)
        self.env_list = maze_group[:self.level_ranges.num_samples]

        # TODO fix this hard-coded part
        for maze in self.env_list:
            maze.exit_reward = 0. # TODO: this is because we are working on memory only right now
            maze.step_reward = 0.
            maze.crash_reward = 0.
            maze.timeout_reward = maze.crash_reward
            maze.timeout_steps = 9 # TODO: same thing--we are making the maze terminate quickly to speed the memory test

        return self.env_list.pop(0)


def activate(queues, params, level_ranges):
    p_name = current_process().name
    print("Activating an actor-critic from {0}.".format(p_name))
    single_ac = AC3Process(queues, params, level_ranges)
    # todo testing
    single_ac.run()
    print("Run in {0} complete.".format(p_name))


def get_group(lowest_lev, highest_lev):
    mazes = []
    for lev in range(lowest_lev, highest_lev + 1):
        print("loading level ", lev)
        filename = "level_" + str(lev) + ".dat"
        with open(filename, "rb") as f:
            maze_group = pickle.load(f)
        mazes += maze_group
    return mazes


EpisodeRecord = namedtuple("EpRecord", ("frame", "field"))


class Recorder(object):
    """
    This tool can be used to record a fixed number of the most recent episodes.
    The primary reason to use this is to allow the agent to test its memory.

    The hope here is that this will stimulate the agent's use of the internal
    field.
    """
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.working_ep = EpisodeRecord([], [])
        self.index = 0
        self.ep_rec = []
        self.batch_ready = False

    def save(self, frame, field):
        self.working_ep.frame.append(frame)
        self.working_ep.field.append(field)

    def finish_episode(self):
        try:
            self.ep_rec[self.index] = self.working_ep
        except IndexError:
            self.ep_rec.append(self.working_ep)
            if len(self.ep_rec) == self.batch_size:
                self.batch_ready = True

        self.index = (self.index + 1) % self.batch_size
        self.working_ep = EpisodeRecord([], [])

    def extract(self):
        if not self.batch_ready:
            return None

        frame_back = []
        frame_fwd = []
        field_fwd = []

        j_back = self.index
        j_fwd = (j_back+1) % self.batch_size
        for _ in range(self.batch_size - 1):
            rec_back = self.ep_rec[j_back]
            rec_fwd = self.ep_rec[j_fwd]
            len_rec_back = len(rec_back.frame)
            len_rec_fwd = len(rec_fwd.frame)

            field_ind_fwd = random.randint(1, len_rec_fwd - 1)
            frame_ind_fwd = random.randint(
                0, min(len_rec_fwd-1, field_ind_fwd))
            frame_ind_back = random.randint(
                0, min(len_rec_back-1, field_ind_fwd))

            field_fwd.append(rec_fwd.field[field_ind_fwd])
            frame_fwd.append(rec_fwd.frame[frame_ind_fwd])
            frame_back.append(rec_back.frame[frame_ind_back])

            j_back = j_fwd
            j_fwd = (j_fwd + 1) % self.batch_size

        # Convert to torch tensors for later input into a discriminator
        field_fwd = torch.stack(field_fwd)
        frame_fwd = torch.stack(frame_fwd)
        frame_back = torch.stack(frame_back)

        return frame_back, frame_fwd, field_fwd


if __name__ == '__main__':
    set_start_method('spawn')

    field_ch = 1
    temp_fr = 2
    # TODO: temp removal of field
    ac_net = networks.ActorCriticNet1F(field_ch, temp_fr, num_hidden=100)

    maze_queue = None  # level ranges mode
    param_queue = Queue()
    glb_t_queue = Queue()

    param_queue.put(ac_net.state_dict())
    glb_t_queue.put(0)

    params = Params(discount=1.,
                    td_steps=5,
                    range=5,
                    field_ch=field_ch,
                    temp_frames=temp_fr,
                    memory_coef=1.,
                    global_t_max=1000000000,
                    entropy_coef=None,
                    recorder_batch_size=100
                    )
    queues = SharedQueues(maze_queue, param_queue, glb_t_queue)

    lev_ranges = LevelRanges(3000, [(20, 20) for _ in range(15)])

    num_processes = 4
    processes = []
    for _ in range(num_processes):
        proc = Process(target=activate, args=(queues, params, lev_ranges))
        processes.append(proc)
    for p in processes:
        p.start()

    for p in processes:
        p.join()

    print("training complete")
    ac_net.load_state_dict(param_queue.get())
    param_queue.put(ac_net.state_dict())
