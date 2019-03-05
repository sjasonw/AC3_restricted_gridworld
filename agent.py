import collections
import random

import gridworld
import tools
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import networks


# import torch.nn.utils.rnn as rnn

# --------------------------------------------------------------------------
# Main agent class
# --------------------------------------------------------------------------

class Agent(object):

    def __init__(self, agent_range, discount=.99,
                 mem_len=10000, sensory_rigidity=5.,
                 internal_ch=1, temporal_frames=4, entropy_coef=.01,
                 memory_coef=.9):
        """

        :param agent_range:
        The agent can see a distance of agent_range in that its view is a
        box around it with width 2 * agent_range + 1

        :param discount:
        Standard discount factor for RL

        :param episode_mem_lim:
        The maximum number of episodes that the agent can store in its log

        :param sensory_rigidity:
        See networks.SensoryNetFixed
        """
        self.environment = None
        self.discount = discount
        self.range = agent_range
        width = 2 * agent_range + 1
        self.view_size = (width, width)
        self.view = torch.zeros(self.view_size)
        self.internal_field_ch = internal_ch
        self.internal_field_size = (internal_ch, 28, 28)
        self.temporal_frames = temporal_frames
        self.entropy_coef = entropy_coef
        self.memory_coef = memory_coef

        self.internal_field = torch.zeros(self.internal_field_size)
        self.mem_capacity = mem_len
        self.history = Memory(capacity=mem_len)

        self.steps = [
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
            np.array([1, 0])
        ]
        # We must make sure that the agent does not attempt to act before
        # updating its view
        self.ready_to_move = False

        self.preprocessor = networks.SensoryNetFixed(sensory_rigidity)
        self.reader_net = networks.ConvolutionalReader(
            internal_ch, temporal_frames, self.range)
        self.value_net = networks.ValueNet(self.range, internal_ch)
        self.target_net = networks.ValueNet(self.range, internal_ch)
        self.policy_net = networks.InternalPolicy(self.range, internal_ch)

        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.value_net.parameters(),
                                    lr=2**-12, betas=(.5, .999))
        self.internal_optimizer = optim.Adam(self.policy_net.parameters(),
                                             lr=2**-8, betas=(.5, .999))

        self.step_counter = 0

    def set_environment(self, maze):

        assert isinstance(maze, gridworld.BoxGame), "Argument maze must" \
                                                    "have type BoxGame."
        assert self.range <= maze.pad_size, "Environment padding size is too" \
                                            " small given the agent's range."
        self.environment = maze
        self.get_view()

    def move(self, move_index):
        """
        :param int move_index: an index for self.steps
        :return: (float reward, bool terminal):
        reward: the reward the environment returns after the action
        terminal: True if the new state is terminal, False otherwise
        """
        assert self.ready_to_move, "Error: Agent.move called before calling" \
                                   "Agent.get_view."
        step = self.steps[move_index]
        (reward, terminal) = self.environment.move_agent(step)
        self.ready_to_move = False
        self.get_view()

        return reward, terminal

    def get_view(self, noise_scale=.01):
        if self.ready_to_move:
            return self.view
        self.view = self.environment.agent_view(self.range)
        self.view += noise_scale * torch.randn(self.view.size())
        self.ready_to_move = True
        return self.view

    def reset_grid(self):
        self.environment.reset_grid()
        self.ready_to_move = False
        self.get_view()

    def reset_internal_field(self):
        self.internal_field = torch.zeros(self.internal_field_size)

    def update_target_net(self):
        """Updates the target network params to match the value network."""
        current_state_dict = self.value_net.state_dict()
        self.target_net.load_state_dict(current_state_dict)

    def e_greedy_episode(self, epsilon, policy_grad, watch=False):
        """

        :param epsilon:

        :param policy_grad: bool
            When True, the actor-critic policy gradient method is used to change
            self.policy_net.

        :param watch:
        """
        show_field = (random.random() < .001)

        # TODO: delete.  This is just for testing
        init_steps = self.step_counter
        reward_list = []

        init_state = self.preprocessor(self.view.unsqueeze(0))
        frames = [init_state for _ in range(self.temporal_frames)]
        frames = torch.stack(frames, dim=1)

        # We are required to first perform an initial action
        with torch.no_grad():
            state = self.reader_net(self.internal_field.unsqueeze(0), frames)
            value = self.value_net(state)

        init_field = self.internal_field.clone()
        distribution = self.policy_net(state)
        change = distribution.sample()
        self.internal_field = self.memory_coef * self.internal_field + \
                              change.squeeze(dim=0)
        self.internal_field.clamp_(-10., 10.)

        greedy_value = value.max()
        greedy_action = tools.fair_argmax(value.view(4))
        a, r, terminal = self.e_greedy_move(epsilon, greedy_action, watch)

        reward_list.append(r)

        # Get the new frame and stack it with the former frames.
        new_frame = self.preprocessor(self.view.unsqueeze(0))
        new_frame.unsqueeze_(dim=1)  # time dimension
        frames = torch.cat((frames, new_frame), dim=1)
        # Record this transition.
        # TODO: find out if cloning is necessary
        fwd_field = self.internal_field.clone()
        self.history.memorize(
            frames[0].clone(),
            torch.stack([init_field, fwd_field]),
            a, r, terminal)
        frames = frames[:,1:]  # prepare for next time step
        init_field = fwd_field

        # Now all the other steps
        discount_factor = 1.
        while not terminal:
            self.step_counter += 1

            distribution = self.policy_net(state)
            change = distribution.sample()
            self.internal_field = self.memory_coef * self.internal_field + \
                                  change.squeeze(dim=0)
            self.internal_field.clamp_(-10., 10.)

            with torch.no_grad():
                # The state is read without gradients--we only optimize
                # the reader when performing Q-learning with a recall.
                state = self.reader_net(
                    self.internal_field.unsqueeze(0), frames
                )
                next_value = self.value_net(state)
                next_greedy_action = tools.fair_argmax(next_value.view(4))
                next_greedy_value = next_value.max()

            if policy_grad:
                # TODO: for testing only:
                if show_field:
                    self.show_internal_field()
                # END

                # We sum the log  of probabilities since the distribution
                # is uncorrelated
                ell_factor = distribution.log_prob(change).sum()
                return_factor = r + self.discount * next_greedy_value
                return_factor = return_factor - greedy_value  # baseline
                entropy_reg = self.entropy_coef * distribution.entropy().sum()
                self.internal_optimizer.zero_grad()
                loss = -discount_factor*return_factor*ell_factor - entropy_reg
                loss.backward()
                self.internal_optimizer.step()
                discount_factor = discount_factor * self.discount

            greedy_value = next_greedy_value
            greedy_action = next_greedy_action

            a, r, terminal = self.e_greedy_move(epsilon, greedy_action, watch)
            # TODO rem
            reward_list.append(r)

            new_frame = self.preprocessor(self.view.unsqueeze(0))
            new_frame.unsqueeze_(dim=1)
            frames = torch.cat((frames, new_frame), dim=1)
            # TODO: find out if cloning is necessary
            fwd_field = self.internal_field.clone()
            self.history.memorize(
                frames[0].clone(),
                torch.stack([init_field, fwd_field]),
                a, r, terminal
            )
            frames = frames[:,1:]  # prepare for next time step
            init_field = fwd_field


        self.reset_grid()
        self.reset_internal_field()

        #TODO rem
        return reward_list

    def e_greedy_move(self, epsilon, policy_action, watch):
        if random.random() > epsilon:
            a = policy_action
        else:
            a = random.choice(range(4))
        if not watch:
            r, terminal = self.move(a)
        else:
            r, terminal = self.show_move(a)

        return a, r, terminal

    def recall_study(self, batch_size, allow_batch_reduction=True):
        """

        :param batch_size:
        :param allow_batch_reduction: bool set to True if we should allow
        batch_size to be decreased to make it no greater than the number
        of elements stored in memory.

        :return:
        """
        # First we check if the batch size requested is consistent with
        # the recorded history.
        num_recorded_transitions = len(self.history)
        assert num_recorded_transitions > 0,\
            "Cannot learn from an empty Agent.history"
        if batch_size > self.mem_capacity:
            print("WARNING: Agent.recall_study called with a requested "
                  "batch size greater than the agent's mem_capacity.  "
                  "The requested batch_size may be reduced if allowed, but "
                  "the requested batch size can never be attained.")
        if allow_batch_reduction:
            while batch_size > num_recorded_transitions:
                batch_size -= 1
        else:
            assert batch_size <= num_recorded_transitions, \
                "recall_study called with a batch size greater than" \
                "the number of recorded transitions in Agent.history. This" \
                "can be allowed by setting param allow_batch_reduction of" \
                "Agent.recall_study to True."

        recall = self.history.remember(batch_size)
        init_state = self.reader_net(recall.init_fields, recall.init_frames)
        # The value_net is used to compute estimated values for each state
        # action pair in the batch.  The values are organized into a 1-tensor.
        values = self.value_net(init_state)
        values = torch.gather(values, 1, recall.actions.view(batch_size, 1))
        values = values.view(batch_size)
        with torch.no_grad():
            # The target network is used to compute the greedy value at the
            # next time step.  Note that this method does not update the
            # target network to match the value network--this should be done
            # with Agent.update_target_net
            fwd_state = self.reader_net(recall.fwd_fields, recall.fwd_frames)
            fwd_values = self.target_net(fwd_state)
        (fwd_values_greedy, _) = fwd_values.max(dim=1)
        fwd_values_greedy = torch.where(
            recall.not_terminal, fwd_values_greedy, torch.zeros(batch_size)
        )
        return_target = recall.rewards + self.discount * fwd_values_greedy
        loss = nn.functional.smooth_l1_loss(values, return_target)

        if random.random() < .01:
            print("loss:", loss.item())


        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)
        self.optimizer.step()

    def show_view(self, get_view=False):

        if get_view:
            self.get_view()

        view = self.view.clone()
        view[self.range, self.range] = 1.

        plt.matshow(view.numpy(), cmap="Greys", vmin=-2., vmax=2.)
        plt.show(block=False)
        plt.pause(.02)

    def show_move(self, direction, show_full_grid=False):

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

        else:
            self.show_view()

        return reward, terminal

    def show_internal_field(self):
        plt.close()
        plt.imshow(self.internal_field[0].numpy(), vmin=-10.,vmax=10.)
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


Transition = collections.namedtuple("Transition",
                                    ("frames",
                                     "field",
                                     "action",
                                     "reward",
                                     "terminal")
                                    )
RecallBatch = collections.namedtuple("RecallBatch",
                                     ("init_frames",
                                      "fwd_frames",
                                      "init_fields",
                                      "fwd_fields",
                                      "actions",
                                      "rewards",
                                      "not_terminal")
                                     )


class Memory(object):

    def __init__(self, capacity):
        # self.memory will store a list of recorded transitions.
        # self.position will monitor the next index in memory to be stored.
        # When memory runs out, position loops back to zero and we overwrite.
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def memorize(self, *args):
        """
        Saves a single transition into self.memory as a Transition object.
        args should have the form
        (frames, field, action, reward, terminal)

        :param args[0] frames
            torch.Tensor size (temporal_frames + 1, 2, width, width)
            temporal_frames refers to the number of time steps that make
            a state.  The extra frame is because this is a record of the
            state before [:-1] and the state after [1:].  The remaining
            dimensions are from the output of an object of type
            networks.SensoryNetFixed

        :param args[1] field
            torch.Tensor size (2, internal field channels, 28, 28)
            This is the internal field of the agent before and
            after the action.  The first index with rank 2 is for
            the before and after field with index 0 and 1 respectively.

        :param args[2] action
            int: represents the action

        :param args[3] reward
            float: the reward returned by the environment after the action

        :param args[4] terminal
            bool: True iff the action led to a terminal state
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def remember(self, batch_size):
        """
        """
        transitions = random.sample(self.memory, batch_size)
        init_fr = [transitions[i].frames[:-1] for i in range(batch_size)]
        fwd_fr = [transitions[i].frames[1:] for i in range(batch_size)]
        init_fields = [transitions[i].field[0] for i in range(batch_size)]
        fwd_fields = [transitions[i].field[1] for i in range(batch_size)]
        actions = [transitions[i].action for i in range(batch_size)]
        rewards = [transitions[i].reward for i in range(batch_size)]
        not_term = [not transitions[i].terminal for i in range(batch_size)]
        recall = RecallBatch(
            init_frames=torch.stack(init_fr),
            fwd_frames=torch.stack(fwd_fr),
            init_fields=torch.stack(init_fields),
            fwd_fields=torch.stack(fwd_fields),
            actions=torch.tensor(actions),
            rewards=torch.tensor(rewards),
            not_terminal=torch.tensor(not_term)
        )
        return recall

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    from time import time
    ag = Agent(5,temporal_frames=4)
    box = gridworld.BoxGame(5, timeout_steps=75)
    ag.set_environment(box)
    t1 = time()
    for _ in range(3):
        ag.e_greedy_episode(.99, False)

    ag.recall_study(2)


