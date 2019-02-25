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

    def __init__(self, agent_range, discount=.99, episode_mem_lim=30,
                 short_term_cap=500, blur=None, sensory_rigidity=5.):
        self.environment = None
        self.discount = discount
        self.range = agent_range
        width = 2 * agent_range + 1
        self.view_size = (width, width)
        self.view = torch.zeros(self.view_size)
        self.blur = blur
        self.short_term_cap = short_term_cap
        self.episode_capacity = episode_mem_lim
        self.history = Memory(capacity=episode_mem_lim)

        self.steps = [
            np.array([0, 1]),
            np.array([-1, 0]),
            np.array([0, -1]),
            np.array([1, 0])
        ]
        self.ready_to_move = False

        self.preprocessor = networks.SensoryNetFixed(sensory_rigidity)
        self.value_net = networks.ActionValueNet(agent_range)
        self.target_net = networks.ActionValueNet(agent_range)
        self.target_net.load_state_dict(self.value_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.value_net.parameters(),
                                       lr=5.e-6,momentum=.99)

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

    def get_view(self, no_blur=False):
        if self.ready_to_move:
            return self.view

        if (self.blur is None) or no_blur:
            self.view = self.environment.agent_view(self.range)

        else:

            blur_factor = tools.perturb_blur(self.blur)

            self.view = (self.environment.agent_view(self.range)
                         + blur_factor*self.view)

        self.ready_to_move = True
        return self.view

    def reset_grid(self):
        self.environment.reset_grid()
        self.ready_to_move = False
        self.get_view(no_blur=True)

    def update_target_net(self):
        """Updates the target network params to match the value network."""
        current_state_dict = self.value_net.state_dict()
        self.target_net.load_state_dict(current_state_dict)

    def e_greedy_episode(self, epsilon, reset=False, watch=False):
        """
        This is the main data-gathering loop for our modified deep Q-learning.
        This method runs through a single episode with no learning during
        the run.  Actions are selected via an epsilon-greedy policy
        defined by self.value_net.  When a terminal state is reached,
        the episode is saved into self.history.  This episode can
        then be used by another method (see self.learn) for training.

        This method should be called after Agent.get_view has been called.
        Note that get_view is automatically called by Agent.set_environment.
        """
        # Begin by initializing a record for frames, hidden LSTM states,
        # actions, and rewards. Note that we set this up for a batch size
        # of 1
        init_state = self.preprocessor(self.view.unsqueeze(0)) # TODO: clone needed??
        frames = [init_state]
        h_0, c_0 = (torch.zeros(self.value_net.hidden_state_shape(1))
                    for _ in range(2))
        hidden = [(h_0, c_0)]

        #hidden = [None]
        actions = []
        rewards = []

        terminal = False
        while not terminal:
            self.step_counter += 1
            value, h_and_c = self.value_net(frames[-1].unsqueeze(1), hidden[-1])
            for hid in h_and_c:
                hid.detach_()

            hidden.append(h_and_c)
            if random.random() > epsilon:
                with torch.no_grad():
                    a = tools.fair_argmax(value.view(4))
            else:
                a = random.choice(range(4))
            if not watch:
                r, terminal = self.move(a)
            else:
                r, terminal = self.show_move(a)
            actions.append(a)
            rewards.append(r)
            frames.append(self.preprocessor(self.view.unsqueeze(0)))
        # !!
        print("normal / crash / exit: ",rewards.count(-.1), " / ", rewards.count(-.2), " / ", rewards.count(1.))
        frames = torch.stack(frames)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)

        # Only record the last self.short_term_cap+1 frames of the episode.
        frames = frames[-(self.short_term_cap + 1):]
        hidden = hidden[-(self.short_term_cap + 1):]
        actions = actions[-self.short_term_cap:]
        rewards = rewards[-self.short_term_cap:]
        self.history.memorize_ep(frames, hidden, actions, rewards)
        if reset:
            self.reset_grid()

    def recall_study(self, batch_size, backup=1, allow_batch_reduction=True):
        """
        :param backup:
        :param batch_size:
        :param allow_batch_reduction: bool set to True if we should allow
        batch_size to be decreased to make it no greater than the number
        of elements stored in memory.

        :return:
        """
        # First we check if the batch size requested is consistent with
        # the recorded history.
        num_recorded_eps = len(self.history)
        assert num_recorded_eps > 0, "Cannot learn from an empty Agent.history"
        if batch_size > self.episode_capacity:
            print("WARNING: Agent.recall_study called with a requested "
                  "batch size greater than the agent's episode_capacity.  "
                  "The requested batch_size may be reduced if allowed, but "
                  "the requested batch size can never be attained.")
        if allow_batch_reduction:
            while batch_size > num_recorded_eps:
                batch_size -= 1
        else:
            assert batch_size <= num_recorded_eps, \
                "recall_study called with a batch size greater than" \
                "the number of recorded episodes in Agent.history. This" \
                "is can be allowed by setting param allow_batch_reduction of" \
                "Agent.recall_study to True."

        recall = self.history.remember(batch_size, backup=backup)
        frames = recall["back_frames"]
        hidden = recall["hidden"]
        actions = recall["actions"]
        rewards = recall["rewards"]
        not_term = recall["not_terminal"]
        fwd_frames = recall["forward_frames"]

        # The value_net is used to compute estimated values for each state
        # action pair in the batch.  The values are organized into a 1-tensor.
        (values, hidden) = self.value_net(frames, hidden)
        values = torch.gather(values, 1, actions.view(batch_size, 1))
        values = values.view(batch_size)

        # The target network is used to compute the greedy value at the
        # next time step.  Note that this method does not update the
        # target network to match the value network--this should be done
        # with Agent.update_target_net
        (fwd_values, _) = self.target_net(fwd_frames.unsqueeze(dim=0), hidden)
        fwd_values.detach_()
        (fwd_values_greedy, _) = fwd_values.max(dim=1)
        fwd_values_greedy = torch.where(
            not_term, fwd_values_greedy, torch.zeros(batch_size)
        )
        return_target = rewards + self.discount * fwd_values_greedy

        loss = nn.functional.smooth_l1_loss(values, return_target)
        #print("loss: ", loss)
        if np.random.random() <.01:
            print("sample values and targets:")
            print(values)
            print(return_target)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=False)
        # TODO: look into doing this
        for param in self.value_net.parameters():
            param.grad.data.clamp_(-10., 10.)

        #print("before step ", self.value_net.parameters().__next__())
        self.optimizer.step()
        #print("after step ", self.value_net.parameters().__next__())




    # def q_learning_experience(self,
    #                           num_episodes,
    #                           rate,
    #                           epsilon,
    #                           batch_size=10,
    #                           environment=None,
    #                           watch=False,
    #                           epsilon_factor=None):
    #
    #     completion_steps = []
    #
    #     if environment:
    #         self.set_environment(environment)
    #     else:
    #         assert self.environment, "error: if the environment is not" \
    #                                  "already set, then you must include" \
    #                                  "it as an argument to Agent." \
    #                                  "q_learning_experience"
    #
    #     if epsilon_factor is None:
    #         def epsilon_factor(x): return 1
    #
    #     opt = optim.SGD(
    #         [
    #             {'params' : self.sensory_net.parameters()},
    #             {'params' : self.value_net.parameters()}
    #         ],
    #         lr=rate, weight_decay=.01
    #     )
    #
    #     sensory_opt = optim.SGD(
    #         self.sensory_net.parameters(), lr=rate, weight_decay=.01
    #     )
    #     value_opt = optim.SGD(
    #         self.value_net.parameters(), lr=rate, weight_decay=.01
    #     )
    #
    #     for episode in range(num_episodes):
    #
    #         terminal = False
    #         t=0
    #
    #         # unsqueeze the view because we will form a sequence of views
    #         sequence = (self.get_view().clone()).unsqueeze(0)
    #
    #         t_start = time()
    #
    #         while not terminal:
    #             t += 1
    #
    #             # select and epsilon-greedy action
    #             action = self.eps_greedy(
    #                 epsilon * epsilon_factor(episode), sequence
    #             )
    #
    #
    #             # act with the selected action, observe the new state
    #             if not watch:
    #                 (reward, terminal) = self.move(action)
    #             else:
    #                 (reward, terminal) = self.show_move(action, True)
    #
    #
    #
    #             # Update the state sequence and memorize SARS
    #             next_state = (self.get_view().clone()).unsqueeze(0)
    #             sequence = torch.cat((sequence, next_state))
    #             if len(sequence) > self.short_term_cap:
    #                 sequence = sequence[1:]
    #
    #             self.experience.memorize(action, reward, sequence, terminal)
    #
    #
    #             objective = self.stochastic_objective(batch_size)
    #
    #             objective.backward()
    #
    #             # sensory_opt.step()
    #
    #             # value_opt.step()
    #
    #
    #             opt.step()
    #             opt.zero_grad()
    #
    #
    #             # sensory_opt.zero_grad()
    #             # value_opt.zero_grad()
    #
    #         print("episode ", episode, " completed in ", t, " steps")
    #         completion_steps.append(t)
    #         self.environment.reset_grid()
    #
    #     return completion_steps
    #
    #
    # def stochastic_objective(self, batch_size):
    #
    #     recall = self.experience.remember(batch_size, adjust_size=True)
    #     (actions, rewards, states, non_terminal) = recall
    #
    #     sensed = self.sensory_net(states.data)
    #
    #     sensed = rnn.PackedSequence(sensed, states.batch_sizes)
    #
    #
    #
    #     rnn_packed_output = self.value_net(sensed)[0]
    #     padded = rnn.pad_packed_sequence(rnn_packed_output)
    #     padded_data = padded[0]
    #     lengths = padded[1]
    #
    #
    #     ### !!!!
    #     padded_data = 10.*padded_data
    #
    #     values = [
    #         padded_data[lengths[j] - 2][j] for j in range(len(lengths))
    #     ]
    #     forward_values = [
    #         padded_data[lengths[j] - 1][j] for j in range(len(lengths))
    #     ]
    #
    #     # We now convert values and forward_values into (batch_size, 4) tensors
    #     values = torch.stack(values)
    #     forward_values = torch.stack(forward_values)
    #
    #     selected_values = [
    #         values[b][actions[b]] for b in range(len(actions))
    #     ]
    #     selected_values = torch.stack(selected_values)
    #
    #     (greedy_values, greedy_actions) = forward_values.max(dim=1)
    #
    #     target = rewards + self.discount * non_terminal * greedy_values
    #     target.detach_()
    #
    #     return tools.huber(target - selected_values)

    # def short_term_process(self, sequence):
    #     """
    #     TODO: change for RNN implementation
    #
    #     Takes a sequence of the form [state, action, state, action, ..., state]
    #     and returns a stack of the last `self.short_term_cap` states.  This is
    #     to accomplish the preprocessing used in deep Q-learning with
    #     experience.  If the short_term_cap is too large for the given sequence,
    #     then the stack will make the fill in the blanks by copying the initial
    #     state as many times as necessary to the top of the output stack.
    #
    #     :param sequence:
    #     Sequence of the form [state, action, state, ..., state] where
    #     each state is a 2d torch tensor and refers to a single frame.
    #     Actions are skipped by this method.
    #
    #     :return:
    #     Returns a rank 3 torch tensor with dimensions
    #     self.short_term_cap X self.view_size
    #     """
    #     num_states = (len(sequence) + 1) // 2
    #     if num_states <= self.short_term_cap:
    #         stack = [
    #             sequence[2 * i]
    #             for i in range(num_states)
    #         ]
    #     else:
    #         stack = [
    #             sequence[-self.short_term_cap + 2*i - 1]
    #             for i in range(self.short_term_cap)
    #         ]
    #
    #     return torch.stack(stack)
    #
    #
    # def eps_greedy(self, epsilon, state):
    #
    #     assert 0 <= epsilon <= 1, "eps_greedy arg epsilon must be in [0,1]."
    #
    #     x = np.random.random()
    #     if x < epsilon:
    #         return np.random.randint(4) # There are 4 directions
    #
    #     else:
    #         with torch.no_grad():
    #             sense_out = self.sensory_net(state) # use sequence times as a batch
    #             sense_out.unsqueeze_(1) # change shape to (times, 1, area dims)
    #
    #             vals = self.value_net(sense_out)[0] #[0] for RNN output extraction
    #             vals = vals[-1][0] # last sequence output, trivial batch
    #
    #             selection = tools.fair_argmax(vals)
    #
    #         return selection
    #
    #
    #
    #
    #
    #
    #
    #
    #         values = self.q_net(state.unsqueeze(0))
    #         if x >.999:
    #             print(values)
    #         return tools.fair_argmax(values)


    # Methods for visualization and testing:

    def show_view(self, get_view=False):

        if get_view:
            self.get_view()

        view = self.view.clone()
        view[self.range,self.range] = 1.

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
        #self.get_view()

        if show_full_grid:
            view = self.view.clone()
            grid = self.environment.grid.clone()

            view[self.range,self.range] = 1.
            grid[tuple(self.environment.agent_point)] = 1.
            tools.show_images(
                [view.numpy(), grid.numpy()],
                titles=["Agent perspective", "Maze map"]
            )

        else:
            self.show_view()

        return reward, terminal

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


EpRecord = collections.namedtuple("EpRecord",
                                  ("frames", "hidden", "actions", "rewards"))


class Memory(object):

    def __init__(self, capacity):
        # self.memory will store a list of recorded episodes (type EpRecord)
        # self.position will monitor the next index in memory to be stored.
        # When memory runs out, position loops back to zero and we overwrite.
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def memorize_ep(self, *args):
        """
        Saves an entire episode into self.memory as an EpRecord object.
        args should have the form
        (frames, hidden, actions, rewards)

        frames: a torch tensor size (sequence len, agent view)
        such that the final item in the sequence is terminal.
        hidden: is a list of hidden states for the RNN that seeds
        it as though the previous frames of the episode were inputted.
        actions and rewards: tensors of size(sequence len - 1)
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = EpRecord(*args)
        self.position = (self.position + 1) % self.capacity

    def remember(self, batch_size, backup):
        """
        For now we implement an sampling that is fair *over episodes*.
        We randomly select batch_size episodes and pick a random
        non-terminal time step t for each of the episodes (t
        depends on the episode).  We then "backup" to the
        time step t-`backup` and output the tuple
        (frames, hidden, action, reward) where
        frames is a list of frames from t-backup to
        t + 1, action is the action that was taken
        at time t, reward is the reward from that action,
        and hidden was the hidden state prior to t-backup.k

        :return:
        A dictionary is returned with keywords
            hidden:
            tuple to input to an RNN with state back_frames which encodes
            information about frames prior to back_frames

            back_frames:
            PackedSequence object that has a collection of sequences
            of frames ending with the frame at time t.

            forward_frames:
            torch.Tensor with shape (batch_size, agent_view). This is the
            batch of states at time t+1.

            actions:
            torch.Tensor with shape (batch_size).  These are actions which
            sent the state at time t to the state at time t+1

            rewards:
            torch.Tensor with shape (batch_size) giving rewards associated
            with the actions selected at time t.

            not_terminal:
            torch.Tensor shape (batch_size) with value 1 if the state at time
            t+1 is not terminal and 0 otherwise.
        """
        permute_me = False
        episodes = random.sample(self.memory, batch_size)
        #print("recorded lens: ", len(episodes[0].frames), len(episodes[0].hidden), len(episodes[0].actions))
        recalled = [None for _ in range(batch_size)]
        for i in range(batch_size):
            size = len(episodes[i].actions)
            index = random.randint(1, size)
            terminal = (index == size)
            if index-1-backup >= 0:
                start = index - 1 - backup
                #print("index", index)
                #print("start", start)
                #print("size", size)
                #print("frame size: ", len(episodes[i].frames))
            else:
                start = 0
                permute_me = True
            frames = episodes[i].frames[start: index+1]
            hidden = episodes[i].hidden[start]
            action = episodes[i].actions[index-1]
            reward = episodes[i].rewards[index-1]
            recalled[i] = {"frames": frames,
                           "hidden": hidden,
                           "action": action,
                           "reward": reward,
                           "terminal": terminal}
        # The function tools.format_recall will convert recalled frames into
        # a packed sequence.  Unfortunately, packed sequences can currently
        # only be made with lists of sequences with descending lengths.  For
        # this reason, call tools.scored_permutation if it is necessary.
        if permute_me:
            #print("scored permutation called")
            lens = [len(recalled[i]["frames"]) for i in range(batch_size)]
            #print(lens)
            recalled = tools.scored_permutation(lens, recalled)
            #print("new lens: ")
            #print([len(recalled[i]["frames"]) for i in range(batch_size)])
        else:
            flag = True
            for i in range(batch_size):
                flag = flag and len(recalled[i]["frames"])== 2+backup
            assert flag, "fail"
        return tools.format_recall(recalled)

    def __len__(self):
        return len(self.memory)




# class Experience2(object):
#     # TODO: the current implementation is very wasteful because entire episodes
#     #   are saved with the form s1, s1 s2, s1 s2 s3, ...  This is a fairly
#     #   easy thing to fix, but can be done later once we know this works.
#
#     def __init__(self, max_mem, state_shape):
#         """
#
#         :param max_mem:
#         :param state_shape: example: (11,11) view
#         """
#         self.max_mem = max_mem
#
#         self.after_states = torch.empty((0,) + state_shape)
#         self.actions = torch.tensor([], dtype=torch.int)
#         self.rewards = torch.tensor([])
#         self.non_terminal = torch.tensor([], dtype=torch.float)
#
#         # self.actions = []
#         # self.rewards = []
#
#         # self.episode_indices will separate distinct eposodes.
#         # the form will be
#         # [[0, b_0], [a_1, b_1], ..., [a_n, b_n]]
#         # with b_0 = a_1, b_1 = a_2, etc.  We also demand that b_n < max_mem
#         # to ensure that memory is not overloaded.
#         self.episode_indices = []
#
#
#     def memorize(self, action, reward, after_state, terminal):
#         """
#
#         :param action:
#         :param reward:
#         :param after_state:
#         torch tensor with shape (sequence_length, agent_view_size)
#
#         :return:
#         """
#         input_length = len(after_state)
#
#         # identify how much memory has been used so far
#         if self.episode_indices == []:
#             used_mem = 0
#         else:
#             used_mem = self.episode_indices[-1][1]
#         remaining = self.max_mem - used_mem
#
#         while input_length > remaining:
#             oldest_size = self.delete_oldest_episode()
#             used_mem -= oldest_size
#             remaining += oldest_size
#
#         self.after_states = torch.cat((self.after_states, after_state))
#
#         action = torch.tensor([action],dtype=torch.int)
#         self.actions = torch.cat(
#             (self.actions, action)
#         )
#         reward = torch.tensor([reward])
#         self.rewards = torch.cat(
#             (self.rewards, reward)
#         )
#         non_terminal = torch.tensor([not terminal], dtype=torch.float)
#         self.non_terminal = torch.cat(
#             (self.non_terminal, non_terminal)
#         )
#
#
#         # self.actions.append(action)
#         # self.rewards.append(reward)
#
#         self.episode_indices.append([used_mem, used_mem + input_length])
#
#     def delete_oldest_episode(self):
#         """
#
#         :return:
#         number of entries deleted or, equivalently, the length of the first
#         episode (which is deleted here).
#         """
#         # del self.actions[0]
#         # del self.rewards[0]
#
#         self.actions = self.actions[1:]
#         self.rewards = self.rewards[1:]
#         self.non_terminal = self.non_terminal[1:]
#
#         oldest_size = self.episode_indices[0][1]
#         self.after_states = self.after_states[oldest_size:]
#
#         self.episode_indices = [
#             [self.episode_indices[i][j] - oldest_size for j in range(2)]
#             for i in range(1, len(self.episode_indices))
#         ]
#
#         return oldest_size
#
#     def remember(self, batch_size, adjust_size=True):
#         """
#
#         :param batch_size:
#         :param adjust_size:
#         :return:
#
#         If adjust_size is False and batch_size is larger
#         than the number of stored episodes,, then None is returned.
#         """
#         num_stored_eps = len(self.episode_indices)
#
#         if not adjust_size and batch_size > num_stored_eps:
#             return None
#
#         while batch_size > num_stored_eps:
#             batch_size -= 1
#
#         batch_selection = np.random.choice(num_stored_eps, batch_size, replace=False)
#
#         batch_selection = self.descending_permutation(batch_selection)
#         #batch_selection = torch.from_numpy(batch_selection)
#         #batch_selection = batch_selection.sort(descending=True)[0]
#         #batch_selection = torch.sort(batch_selection, descending=True)[0]
#
#         actions = self.actions[batch_selection]
#
#         rewards = self.rewards[batch_selection]
#         #ep_indices = self.episode_indices[batch_selection]
#
#         ep_indices = [self.episode_indices[i] for i in batch_selection]
#
#
#         non_terminal = self.non_terminal[batch_selection]
#
#         states = [
#             self.after_states[ep_indices[i][0]: ep_indices[i][1]]
#             for i in range(batch_size)
#         ]
#         packed_states = torch.nn.utils.rnn.pack_sequence(states)
#
#         return actions, rewards, packed_states, non_terminal
#
#     def descending_permutation(self, batch_selection):
#         """
#         TODO: explain this thing or find a better way
#         :param indices:
#         :return:
#         """
#         batch_selection = torch.from_numpy(batch_selection)
#         lengths = [
#             self.episode_indices[i][1]-self.episode_indices[i][0] for i in batch_selection
#         ]
#
#         lengths = torch.tensor(lengths)
#         perm = lengths.sort(descending=True)[1]
#
#         return batch_selection[perm]











###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


#######
##
##
## OLD VERSION
##
##
#######

#
# # --------------------------------------------------------------------------
# Main agent class
# --------------------------------------------------------------------------
#
# class Agent(object):
#
#     def __init__(self, agent_range, discount=.9, exp_mem_limit=10000, short_term_cap=1000, blur=None):
#
#         self.environment = None
#         self.discount = discount
#
#         self.range = agent_range
#         width = 2 * agent_range + 1
#         self.view_size = (width, width)
#         self.view = torch.zeros(self.view_size)
#         self.blur = blur
#
#         self.short_term_cap = short_term_cap
#
#         self.experience = Experience2(exp_mem_limit, self.view_size)
#
#         self.steps = [
#             np.array([0, 1]),
#             np.array([-1, 0]),
#             np.array([0, -1]),
#             np.array([1, 0])
#         ]
#
#         self.ready_to_move = False
#
#         # Network for computing the action-value function
#         # self.q_net = ActionValueNetworkConv(
#         #     self.range,
#         #     min_output=-1.1, # TODO: remove hard-coded numbers
#         #     max_output=1.1
#         # )
#         #
#         self.sensory_net = networks.SensoryNet(agent_range)
#         self.value_net = networks.ValueRNN(agent_range)
#
#     def set_environment(self, maze):
#
#         assert isinstance(maze, gridworld.BoxGame), "Argument maze must" \
#                                                     "have type BoxGame."
#         self.environment = maze
#
#         assert self.range <= maze.pad_size, "Environment padding size is too" \
#                                             " small given the agent's range."
#
#     def move(self, move_index):
#         """
#
#         :param int move_index: an index for self.steps
#
#         :return: (float reward, bool terminal):
#         reward: the reward the environment returns after the action
#         terminal: True if the new state is terminal, False otherwise
#         """
#
#         assert self.ready_to_move, "Error: Agent.move called before calling" \
#                                    "Agent.get_view."
#
#         step = self.steps[move_index]
#         (reward, terminal) = self.environment.move_agent(step)
#
#         self.ready_to_move = False
#
#         return reward, terminal
#
#     def get_view(self):
#         # TODO: remove the central 1
#
#         if self.blur is None:
#             self.view = self.environment.agent_view(self.range)
#
#         else:
#
#             blur_factor = tools.perturb_blur(self.blur)
#
#             self.view = (self.environment.agent_view(self.range)
#                          + blur_factor*self.view)
#
#         self.ready_to_move = True
#         return self.view
#
#     # Q-learning :
#     #
#     # def q_learning_experience(self,
#     #                           num_episodes,
#     #                           rate,
#     #                           epsilon,
#     #                           batch_size=10,
#     #                           environment=None,
#     #                           watch=False):
#     #
#     #     if environment:
#     #         self.set_environment(environment)
#     #     else:
#     #         assert self.environment, "error: if the environment is not" \
#     #                                  "already set, then you must include" \
#     #                                  "it as an argument to Agent." \
#     #                                  "q_learning_experience"
#     #
#     #     # old
#     #     #optimizer = optim.SGD(self.q_net.parameters(),
#     #     #                      lr=rate,momentum=.9, weight_decay=.01)
#     #
#     #     sensory_opt = optim.SGD(
#     #         self.sensory_net.parameters(), lr=rate, weight_decay=.01
#     #     )
#     #     value_opt = optim.SGD(
#     #         self.value_net.parameters(), lr=rate, weight_decay=.01
#     #     )
#     #
#     #
#     #
#     #
#     #
#     #     for episode in range(num_episodes):
#     #         t = 0
#     #         terminal = False
#     #         sequence = [self.get_view().clone()]  # TODO: is cloning necessary?
#     #         processed_1 = self.short_term_process(sequence)
#     #         while not terminal:
#     #
#     #             t += 1
#     #             action = self.eps_greedy(epsilon / (1 + episode)**.5, processed_1)
#     #
#     #             if not watch:
#     #                 (reward, terminal) = self.move(action)
#     #             else:
#     #                 (reward, terminal) = self.show_move(action, True)
#     #
#     #             next_state = self.get_view().clone()  # TODO: cloning?
#     #             sequence += [action, next_state]
#     #             processed_2 = self.short_term_process(sequence)
#     #
#     #             if processed_1 is not None:
#     #                 #self.experience.memorize(processed_1, action, reward, processed_2)
#     #                 self.experience.memorize(action, reward,processed_2)
#     #
#     #             if len(self.experience.episode_indices) >= batch_size:
#     #
#     #                 recall = self.experience.remember(batch_size, adjust_size=False)
#     #                 (a_batch, r_batch, s_pack) = recall
#     #
#     #                 sensed = self.sensory_net(s_pack.data)
#     #                 sensed = rnn.PackedSequence(sensed, s_pack.batch_sizes)
#     #
#     #                 rnn_packed_output = self.value_net(sensed)[0]
#     #                 padded = rnn.pad_packed_sequence(rnn_packed_output)
#     #                 padded_data = padded[0]
#     #                 lengths = padded[1]
#     #
#     #                 last_elements = [padded_data[lengths[j] - 1][j]
#     #                                  for j in range(len(lengths))]
#     #
#     #                 last_elements = torch.tensor(last_elements)
#     #                 (greedy_values, greedy_actions) = last_elements.max(dim=1)
#     #
#     #                 target = r_batch
#     #                 if not terminal:
#     #                     target += self.discount * forward_value_greedy
#     #
#     #                 # make target a constant independent of weights
#     #                 target.detach_()
#     #
#     #                 curr_value = (self.q_net(s_1_batch))[:,action]
#     #                 objective = .5 * (target - curr_value)**2
#     #                 objective = objective.mean()
#     #                 #print("\n\n\nbefore ", objective)
#     #
#     #                 objective.backward()
#     #                 optimizer.step()
#     #                 optimizer.zero_grad()
#     #
#     #
#     #                 ## TEMP:
#     #                 curr_value = (self.q_net(s_1_batch))[:, action]
#     #                 af_objective = .5 * (target - curr_value)**2
#     #                 af_objective = af_objective.mean()
#     #                 #print("decreased?: ", (af_objective <= objective).item() == 1)
#     #                 ## end temp
#     #
#     #             processed_1 = processed_2
#     #
#     #         print("episode ", episode, " completed in ", t, " steps")
#     #         self.environment.reset_grid()
#
#     def q_learning_experience(self,
#                               num_episodes,
#                               rate,
#                               epsilon,
#                               batch_size=10,
#                               environment=None,
#                               watch=False,
#                               epsilon_factor=None):
#
#         completion_steps = []
#
#         if environment:
#             self.set_environment(environment)
#         else:
#             assert self.environment, "error: if the environment is not" \
#                                      "already set, then you must include" \
#                                      "it as an argument to Agent." \
#                                      "q_learning_experience"
#
#         if epsilon_factor is None:
#             def epsilon_factor(x): return 1
#
#         opt = optim.RMSprop(
#             [
#                 {'params' : self.sensory_net.parameters()},
#                 {'params' : self.value_net.parameters()}
#             ]#,
#            # lr=rate, weight_decay=.01
#         )
#
#         sensory_opt = optim.SGD(
#             self.sensory_net.parameters(), lr=rate, weight_decay=.01
#         )
#         value_opt = optim.SGD(
#             self.value_net.parameters(), lr=rate, weight_decay=.01
#         )
#
#         for episode in range(num_episodes):
#
#             terminal = False
#             t=0
#
#             # unsqueeze the view because we will form a sequence of views
#             sequence = (self.get_view().clone()).unsqueeze(0)
#
#             t_start = time()
#
#             while not terminal:
#                 t += 1
#
#                 # select and epsilon-greedy action
#                 action = self.eps_greedy(
#                     epsilon * epsilon_factor(episode), sequence
#                 )
#
#
#                 # act with the selected action, observe the new state
#                 if not watch:
#                     (reward, terminal) = self.move(action)
#                 else:
#                     (reward, terminal) = self.show_move(action, True)
#
#
#
#                 # Update the state sequence and memorize SARS
#                 next_state = (self.get_view().clone()).unsqueeze(0)
#                 sequence = torch.cat((sequence, next_state))
#                 if len(sequence) > self.short_term_cap:
#                     sequence = sequence[1:]
#
#                 self.experience.memorize(action, reward, sequence, terminal)
#
#
#                 objective = self.stochastic_objective(batch_size)
#
#                 objective.backward()
#
#                 # sensory_opt.step()
#
#                 # value_opt.step()
#
#
#                 opt.step()
#                 opt.zero_grad()
#
#
#                 # sensory_opt.zero_grad()
#                 # value_opt.zero_grad()
#
#             print("episode ", episode, " completed in ", t, " steps")
#             completion_steps.append(t)
#             self.environment.reset_grid()
#
#         return completion_steps
#
#
#     def stochastic_objective(self, batch_size):
#
#         recall = self.experience.remember(batch_size, adjust_size=True)
#         (actions, rewards, states, non_terminal) = recall
#
#         sensed = self.sensory_net(states.data)
#
#         sensed = rnn.PackedSequence(sensed, states.batch_sizes)
#
#
#
#         rnn_packed_output = self.value_net(sensed)[0]
#         padded = rnn.pad_packed_sequence(rnn_packed_output)
#         padded_data = padded[0]
#         lengths = padded[1]
#
#
#         ### !!!!
#         padded_data = 10.*padded_data
#
#         values = [
#             padded_data[lengths[j] - 2][j] for j in range(len(lengths))
#         ]
#         forward_values = [
#             padded_data[lengths[j] - 1][j] for j in range(len(lengths))
#         ]
#
#         # We now convert values and forward_values into (batch_size, 4) tensors
#         values = torch.stack(values)
#         forward_values = torch.stack(forward_values)
#
#         selected_values = [
#             values[b][actions[b]] for b in range(len(actions))
#         ]
#         selected_values = torch.stack(selected_values)
#
#         (greedy_values, greedy_actions) = forward_values.max(dim=1)
#
#         target = rewards + self.discount * non_terminal * greedy_values
#         target.detach_()
#
#         return tools.huber(target - selected_values)
#
#     def short_term_process(self, sequence):
#         """
#         TODO: change for RNN implementation
#
#         Takes a sequence of the form [state, action, state, action, ..., state]
#         and returns a stack of the last `self.short_term_cap` states.  This is
#         to accomplish the preprocessing used in deep Q-learning with
#         experience.  If the short_term_cap is too large for the given sequence,
#         then the stack will make the fill in the blanks by copying the initial
#         state as many times as necessary to the top of the output stack.
#
#         :param sequence:
#         Sequence of the form [state, action, state, ..., state] where
#         each state is a 2d torch tensor and refers to a single frame.
#         Actions are skipped by this method.
#
#         :return:
#         Returns a rank 3 torch tensor with dimensions
#         self.short_term_cap X self.view_size
#         """
#         num_states = (len(sequence) + 1) // 2
#         if num_states <= self.short_term_cap:
#             stack = [
#                 sequence[2 * i]
#                 for i in range(num_states)
#             ]
#         else:
#             stack = [
#                 sequence[-self.short_term_cap + 2*i - 1]
#                 for i in range(self.short_term_cap)
#             ]
#
#         return torch.stack(stack)
#
#     # def short_term_process(self, sequence):
#     #     """
#     #     ** OLD VERSION **
#     #
#     #     Takes a sequence of the form [state, action, state, action, ..., state]
#     #     and returns a stack of the last `self.short_term_cap` states.  This is
#     #     to accomplish the preprocessing used in deep Q-learning with
#     #     experience.  If the short_term_cap is too large for the given sequence,
#     #     then the stack will make the fill in the blanks by copying the initial
#     #     state as many times as necessary to the top of the output stack.
#     #
#     #     :param sequence:
#     #     Sequence of the form [state, action, state, ..., state] where
#     #     each state is a 2d torch tensor and refers to a single frame.
#     #     Actions are skipped by this method.
#     #
#     #     :return:
#     #     Returns a rank 3 torch tensor with dimensions
#     #     self.short_term_cap X self.view_size
#     #     """
#     #     stack = torch.empty((self.short_term_cap,) + self.view_size)
#     #     num_missing = self.short_term_cap - (len(sequence) + 1)//2
#     #
#     #     if num_missing <= 0:
#     #         num_missing = 0
#     #
#     #     for i in range(0, num_missing):
#     #         stack[i] = sequence[0]
#     #
#     #     for i in range(num_missing, self.short_term_cap):
#     #         stack[i] = sequence[1 - 2*(self.short_term_cap - i)]
#     #
#     #     return stack
#
#     # def q_learning_static_env(self,
#     #                           num_episodes,
#     #                           rate,
#     #                           epsilon,
#     #                           environment=None,
#     #                           watch=False):
#     #
#     #     if environment:
#     #         self.set_environment(environment)
#     #     else:
#     #         assert self.environment, "error: if the environment is not" \
#     #                                  "already set, then you must include" \
#     #                                  "it as an argument to Agent." \
#     #                                  "q_learning_static_env"
#     #
#     #     optimizer = optim.SGD(self.q_net.parameters(), lr=rate,weight_decay=.03)
#     #     #optimizer = optim.Adam(self.q_net.parameters(),lr=rate, betas=(.5, .999),
#     #     #                       weight_decay=5.)
#     #     optimizer.zero_grad()
#     #
#     #     for episode in range(num_episodes):
#     #         terminal = False
#     #         t=0
#     #         while not terminal:
#     #             t += 1
#     #             self.get_view()
#     #             state = self.view.clone()
#     #             action = self.eps_greedy(epsilon/(1 + episode)**.5)
#     #             if not watch:
#     #                 (reward, terminal) = self.move(action)
#     #             else:
#     #                 (reward, terminal) = self.show_move(action, True)
#     #             self.get_view()
#     #             next_state = self.view
#     #
#     #             target = reward
#     #             if not terminal:
#     #                 target += self.discount * (self.q_net(next_state)).max()
#     #
#     #             #target = reward + self.discount * (self.q_net(next_state)).max()
#     #             value = (self.q_net(state)).flatten()[action]
#     #             objective = .5 * (value - target)**2
#     #
#     #             objective.backward()
#     #             optimizer.step()
#     #             optimizer.zero_grad()
#     #
#     #             #print(self.q_net(state))
#     #             if terminal:
#     #                 print("ending episode: t = ", t)
#     #
#     #             # Q(s, a) += alpha * ( R + max_{a'} \gamma Q(s', a') - Q(s,a) )
#     #
#     #         # this is why this method has the word static in its name:
#     #         self.environment.reset_grid()
#
#     def eps_greedy(self, epsilon, state):
#
#         assert 0 <= epsilon <= 1, "eps_greedy arg epsilon must be in [0,1]."
#
#         x = np.random.random()
#         if x < epsilon:
#             return np.random.randint(4) # There are 4 directions
#
#         else:
#             with torch.no_grad():
#                 sense_out = self.sensory_net(state) # use sequence times as a batch
#                 sense_out.unsqueeze_(1) # change shape to (times, 1, area dims)
#
#                 vals = self.value_net(sense_out)[0] #[0] for RNN output extraction
#                 vals = vals[-1][0] # last sequence output, trivial batch
#
#                 selection = tools.fair_argmax(vals)
#
#             return selection
#
#
#
#
#
#
#
#
#             values = self.q_net(state.unsqueeze(0))
#             if x >.999:
#                 print(values)
#             return tools.fair_argmax(values)
#
#
#     # Methods for visualization and testing:
#
#     def show_view(self, get_view=False):
#
#         if get_view:
#             self.get_view()
#
#         view = self.view.clone()
#         view[self.range,self.range] = 1.
#
#         plt.matshow(view.numpy(), cmap="Greys", vmin=-2., vmax=2.)
#         plt.show(block=False)
#         plt.pause(.02)
#
#     def show_move(self, direction, show_full_grid=False):
#
#         if isinstance(direction, str):
#             key_map = {
#                 "W": 1,
#                 "A": 2,
#                 "S": 3,
#                 "D": 0,
#                 "w": 1,
#                 "a": 2,
#                 "s": 3,
#                 "d": 0
#             }
#             direction = key_map[direction]
#
#         plt.close()
#         (reward, terminal) = self.move(direction)
#         self.get_view()
#
#         if show_full_grid:
#             view = self.view.clone()
#             grid = self.environment.grid.clone()
#
#             view[self.range,self.range] = 1.
#             grid[tuple(self.environment.agent_point)] = 1.
#             tools.show_images(
#                 [view.numpy(), grid.numpy()],
#                 titles=["Agent perspective", "Maze map"]
#             )
#
#         else:
#             self.show_view()
#
#         return reward, terminal
#
#     def play(self, show_grid=False, discount=.95):
#
#         self.show_view(get_view=True)
#
#         key_map = {
#             "W": 1,
#             "A": 2,
#             "S": 3,
#             "D": 0,
#             "w": 1,
#             "a": 2,
#             "s": 3,
#             "d": 0
#         }
#
#         t = 0
#         total_return = 0
#         terminal = False
#
#         while not terminal:
#
#             action = input("Use WASD to move (you have to push enter--sorry).")
#             try:
#                 (reward, terminal) = self.show_move(key_map[action],
#                                                      show_full_grid=show_grid)
#                 total_return += discount ** t * reward
#                 t += 1
#
#             except KeyError:
#                 print("input error")
#
#         print("END! Final return is", total_return)
#
#
# class Experience2(object):
#     # TODO: the current implementation is very wasteful because entire episodes
#     #   are saved with the form s1, s1 s2, s1 s2 s3, ...  This is a fairly
#     #   easy thing to fix, but can be done later once we know this works.
#
#     def __init__(self, max_mem, state_shape):
#         """
#
#         :param max_mem:
#         :param state_shape: example: (11,11) view
#         """
#         self.max_mem = max_mem
#
#         self.after_states = torch.empty((0,) + state_shape)
#         self.actions = torch.tensor([], dtype=torch.int)
#         self.rewards = torch.tensor([])
#         self.non_terminal = torch.tensor([], dtype=torch.float)
#
#         # self.actions = []
#         # self.rewards = []
#
#         # self.episode_indices will separate distinct eposodes.
#         # the form will be
#         # [[0, b_0], [a_1, b_1], ..., [a_n, b_n]]
#         # with b_0 = a_1, b_1 = a_2, etc.  We also demand that b_n < max_mem
#         # to ensure that memory is not overloaded.
#         self.episode_indices = []
#
#
#     def memorize(self, action, reward, after_state, terminal):
#         """
#
#         :param action:
#         :param reward:
#         :param after_state:
#         torch tensor with shape (sequence_length, agent_view_size)
#
#         :return:
#         """
#         input_length = len(after_state)
#
#         # identify how much memory has been used so far
#         if self.episode_indices == []:
#             used_mem = 0
#         else:
#             used_mem = self.episode_indices[-1][1]
#         remaining = self.max_mem - used_mem
#
#         while input_length > remaining:
#             oldest_size = self.delete_oldest_episode()
#             used_mem -= oldest_size
#             remaining += oldest_size
#
#         self.after_states = torch.cat((self.after_states, after_state))
#
#         action = torch.tensor([action],dtype=torch.int)
#         self.actions = torch.cat(
#             (self.actions, action)
#         )
#         reward = torch.tensor([reward])
#         self.rewards = torch.cat(
#             (self.rewards, reward)
#         )
#         non_terminal = torch.tensor([not terminal], dtype=torch.float)
#         self.non_terminal = torch.cat(
#             (self.non_terminal, non_terminal)
#         )
#
#
#         # self.actions.append(action)
#         # self.rewards.append(reward)
#
#         self.episode_indices.append([used_mem, used_mem + input_length])
#
#     def delete_oldest_episode(self):
#         """
#
#         :return:
#         number of entries deleted or, equivalently, the length of the first
#         episode (which is deleted here).
#         """
#         # del self.actions[0]
#         # del self.rewards[0]
#
#         self.actions = self.actions[1:]
#         self.rewards = self.rewards[1:]
#         self.non_terminal = self.non_terminal[1:]
#
#         oldest_size = self.episode_indices[0][1]
#         self.after_states = self.after_states[oldest_size:]
#
#         self.episode_indices = [
#             [self.episode_indices[i][j] - oldest_size for j in range(2)]
#             for i in range(1, len(self.episode_indices))
#         ]
#
#         return oldest_size
#
#     def remember(self, batch_size, adjust_size=True):
#         """
#
#         :param batch_size:
#         :param adjust_size:
#         :return:
#
#         If adjust_size is False and batch_size is larger
#         than the number of stored episodes,, then None is returned.
#         """
#         num_stored_eps = len(self.episode_indices)
#
#         if not adjust_size and batch_size > num_stored_eps:
#             return None
#
#         while batch_size > num_stored_eps:
#             batch_size -= 1
#
#         batch_selection = np.random.choice(num_stored_eps, batch_size, replace=False)
#
#         batch_selection = self.descending_permutation(batch_selection)
#         #batch_selection = torch.from_numpy(batch_selection)
#         #batch_selection = batch_selection.sort(descending=True)[0]
#         #batch_selection = torch.sort(batch_selection, descending=True)[0]
#
#         actions = self.actions[batch_selection]
#
#         rewards = self.rewards[batch_selection]
#         #ep_indices = self.episode_indices[batch_selection]
#
#         ep_indices = [self.episode_indices[i] for i in batch_selection]
#
#
#         non_terminal = self.non_terminal[batch_selection]
#
#         states = [
#             self.after_states[ep_indices[i][0]: ep_indices[i][1]]
#             for i in range(batch_size)
#         ]
#         packed_states = torch.nn.utils.rnn.pack_sequence(states)
#
#         return actions, rewards, packed_states, non_terminal
#
#     def descending_permutation(self, batch_selection):
#         """
#         TODO: explain this thing or find a better way
#         :param indices:
#         :return:
#         """
#         batch_selection = torch.from_numpy(batch_selection)
#         lengths = [
#             self.episode_indices[i][1]-self.episode_indices[i][0] for i in batch_selection
#         ]
#
#         lengths = torch.tensor(lengths)
#         perm = lengths.sort(descending=True)[1]
#
#         return batch_selection[perm]
#
#
#
#
#
# class Experience(object):
#
#     def __init__(self, max_mem, state_shape):
#         self.state_1 = torch.empty((max_mem,) + state_shape)
#         self.action = torch.empty(max_mem, dtype=torch.int)
#         self.reward = torch.empty(max_mem)
#         self.state_2 = torch.empty((max_mem,) + state_shape)
#
#         self.length = 0
#         self.max_mem = max_mem
#
#     def memorize(self, s_1, a, r, s_2):
#         if self.length >= self.max_mem:
#
#             self.state_1 = torch.cat(
#                 (self.state_1[1:], s_1.unsqueeze(0))
#             )
#             self.action = torch.cat(
#                 (self.action[1:], torch.tensor([a], dtype=torch.int))
#             )
#             self.reward = torch.cat(
#                 (self.reward[1:], torch.tensor([r], dtype=torch.float))
#             )
#             self.state_2 = torch.cat(
#                 (self.state_2[1:], s_2.unsqueeze(0))
#             )
#
#         else:
#             self.state_1[self.length] = s_1
#             self.action[self.length] = torch.tensor(a) # a is a numpy int so this is needed
#             self.reward[self.length] = r
#             self.state_2[self.length] = s_2
#
#             self.length += 1
#
#     def remember(self, batch_size):
#
#         assert batch_size <= self.length, "error: object of type Experience" \
#                                           "cannot call `remember` with a" \
#                                           "batch size greater than the" \
#                                           "number of items stored in memory"
#
#         batch_ind = np.random.choice(self.length, batch_size, replace=False)
#
#         state_1 = self.state_1[batch_ind]
#         action = self.action[batch_ind]
#         reward = self.reward[batch_ind]
#         state_2 = self.state_2[batch_ind]
#
#         return state_1, action, reward, state_2
#

# --------------------------------------------------------------------------
# Artificial neural network for value function approximation
# --------------------------------------------------------------------------
#
# class ActionValueNetworkConv(nn.Module):
#     """
#     Right now this only works for agents with range 2!!
#     TODO: generalize to arbitrary range.
#     """
#
#     def __init__(self, agent_range, min_output=-4., max_output=4.):
#
#         super(ActionValueNetworkConv, self).__init__()
#
#         self.agent_range = agent_range
#         self.num_in_features = (2*agent_range + 1)**2
#         self.max_output = max_output
#         self.min_output = min_output
#
#         self.num_features_1 = 10
#         self.num_features_2 = 20
#
#         self.convolutional_layers = nn.Sequential(
#
#             nn.Conv2d(1,self.num_features_1,3),
#             nn.ReLU(),
#             nn.Conv2d(self.num_features_1,self.num_features_2, 2),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#
#         )
#
#         self.linear_layers = nn.Sequential(
#             nn.Linear(16 * self.num_features_2, 40),
#             nn.ReLU(),
#             nn.Dropout(p=.5),
#             nn.Linear(40,4),
#             nn.Tanh()
#         )
#
#     def forward(self, agent_view):
#
#         agent_view[0,0, self.agent_range, self.agent_range] = 0.
#
#         features = self.convolutional_layers(agent_view.view(-1,1,11,11))
#
#         features_flat = features.view(-1, tools.num_flat_features(features))
#
#         out = self.linear_layers(features_flat)
#
#         b = self.max_output
#         a = self.min_output
#         return out * (b - a)/2. + (a + b)/2.
#
#
# class SemiConv(nn.Module):
#     """
#     The point of this
#     """
#
#     def __init__(self, agent_range, in_stack_size, min_output=-1., max_output=1.):
#
#         super(SemiConv, self).__init__()
#
#         self.num_in_features = (2*agent_range + 1)**2
#         self.max_output = max_output
#         self.min_output = min_output
#         self.agent_range = agent_range
#         self.in_stack_size = in_stack_size
#
#         self.feature_network = nn.Sequential(
#             nn.Linear(self.num_in_features, 40),  # TODO: work on architecture
#             nn.ReLU(),
#             nn.Dropout(p=.5),
#             nn.Linear(40,20),
#             nn.ReLU(),
#             nn.Dropout(p=.5)
#         )
#
#         self.analysis = nn.Sequential(
#             nn.Linear(20 * self.in_stack_size, 4),
#             nn.Hardtanh(min_val=-10., max_val=10.)
#         )
#
#     def forward(self, agent_view):
#
#         #agent_view[:,self.agent_range, self.agent_range] = ... TODO: fix this annoyance
#
#         # out = agent_view.view(-1,(2*agent_range + 1)**2 )
#         #
#         # out = self.network(out)
#
#         out = agent_view.view(-1, self.num_in_features)
#         out = self.feature_network(out)
#
#         out = out.view(-1,self.in_stack_size, 20)
#         out = out.view(-1,self.in_stack_size * 20)
#
#         out = self.analysis(out)
#
#         #b = self.max_output
#         #a = self.min_output
#         #return out * (b - a)/2. + (a + b)/2.
#         return out
#
#
# class ActionValueNetwork(nn.Module):
#     """
#     The point of this
#     """
#
#     def __init__(self, agent_range, in_stack_size, min_output=-1., max_output=1.):
#
#         super(ActionValueNetwork, self).__init__()
#
#
#         self.num_in_features = in_stack_size * (2*agent_range + 1)**2
#         self.max_output = max_output
#         self.min_output = min_output
#         self.agent_range = agent_range
#         self.in_stack_size = in_stack_size
#
#         self.network = nn.Sequential(
#             nn.Linear(self.num_in_features, 60),  # TODO: work on architecture
#             nn.ReLU(),
#             nn.Dropout(p=.5),
#             nn.Linear(60,4),
#             nn.Tanh()
#         )
#
#     def forward(self, agent_view):
#
#         #agent_view[:,self.agent_range, self.agent_range] = ... TODO: fix this annoyance
#
#         # out = agent_view.view(-1,(2*agent_range + 1)**2 )
#         #
#         # out = self.network(out)
#
#         out = agent_view.view(-1, self.num_in_features)
#         out = self.network(out)
#
#         b = self.max_output
#         a = self.min_output
#         return out * (b - a)/2. + (a + b)/2.
#

if __name__ == "__main__":
    # agent_range = 5
    # game = gridworld.BoxGame(10,block_rate=0.005,num_exits=4, num_pits=0,
    #                          num_empty_targets=0,watch_construction=False,
    #                          max_tree_steps=100,bias=.95,pad_size=agent_range,
    #                          instant_death_p=None)
    # agent = Agent(agent_range, episode_mem_lim=30)
    # agent.set_environment(game)
    #
    # agent.e_greedy_episode(epsilon=.9, reset=True)

    def epsilon_control(ep):
        if ep < 100:
            return .9
        elif ep < 200:
            return .8
        elif ep < 300:
            return .75
        elif ep < 500:
            return .7
        elif ep < 700:
            return .6
        elif ep < 1000:
            return .5
        elif ep < 1500:
            return .4
        elif ep < 2000:
            return .3

    def learn_num(ep):
        if ep < 10:
            return 0
        elif ep < 50:
            return 1
        elif ep < 100:
            return 3
        elif ep < 300:
            return 5
        elif ep < 600:
            return 10
        else:
            return 25

    from time import time
    agent_range = 5
    game = gridworld.BoxGame(10,block_rate=0.005,num_exits=4, num_pits=0,
                             num_empty_targets=0,watch_construction=False,
                             max_tree_steps=100,bias=.95,pad_size=agent_range,
                             instant_death_p=None)



    agent = Agent(agent_range, episode_mem_lim=30)
    agent.set_environment(game)

    for ep in range(1000):
        epsilon = epsilon_control(ep)
        print("starting episode ", ep, " with epsilon = ", epsilon)

        if ep % 100 == 99 and ep > 500:
            print("showing episode with epsilon = ", epsilon, end="")
            agent.e_greedy_episode(epsilon, watch=True, reset=True)
            print("done")
        else:
            agent.e_greedy_episode(epsilon, watch=False, reset=True)

        for _ in range(learn_num(ep)):
            agent.recall_study(5, backup=1, allow_batch_reduction=True)
        if ep % 2 == 1:
            agent.update_target_net()

        print("making new grid...", end="")
        game = gridworld.BoxGame(10, block_rate=0.005, num_exits=4, num_pits=0,
                                 num_empty_targets=0, watch_construction=False,
                                 max_tree_steps=100, bias=.95,
                                 pad_size=agent_range,
                                 instant_death_p=None)
        agent.set_environment(game)
        print("done")


    #
    #
    # print("initial gathering")
    # t1 = time()
    # for i in range(10):
    #     agent.e_greedy_episode(1., reset=True)
    #     agent.environment.reset_grid() #?
    #
    # t2 = time()
    #
    # print("first study")
    #
    #
    # for j in range(2):
    #     print(j)
    #
    #     agent.recall_study(3, backup=1, allow_batch_reduction=True)
    #     agent.update_target_net()
    #
    # t3 = time()
    #
    # print("entering training loop:")
    # for ep in range(1000):
    #     print("ep ", ep)
    #     if ep % 100 == 99 and ep > 500:
    #         agent.e_greedy_episode(1. / (1 + ep) ** .15, watch=True, reset=True)
    #     else:
    #         agent.e_greedy_episode(1. / (1 + ep) ** .15, watch=False, reset=True)
    #
    #     agent.recall_study(5, backup=1, allow_batch_reduction=True)
    #     if ep % 2 == 1:
    #         agent.update_target_net()







    #
    #
    # info = "IDP .005, lr 1.0 (single opt), high epsilon, batch_size 10"
    #
    # rate = .005
    # agent_range = 5
    # gamma = .99
    # agent = Agent(
    #     agent_range=agent_range,
    #     discount=gamma,
    #     exp_mem_limit=10000,
    #     short_term_cap=20,
    #     blur=None
    # )
    # step_count_list = []
    #
    # def interpolate(x, y1, y2, xmin, xmax):
    #
    #     if x< xmin:
    #         return y1
    #     elif x > xmax:
    #         return y2
    #     else:
    #         q = (x - xmin)/(xmax - xmin)
    #         return y1*(1-q) + y2*q
    #
    #
    #
    # for env_count in range(1000):
    #     print("environment # :", env_count)
    #
    #     size = 12
    #     block = 0
    #     exits=10
    #
    #     print("building maze...",end='')
    #     game = gridworld.BoxGame(size,
    #                              block_rate=block,
    #                              num_exits=exits,
    #                              num_pits=0,
    #                              num_empty_targets=0,
    #                              watch_construction=False,
    #                              max_tree_steps=200,
    #                              bias=.9,
    #                              pad_size=agent_range,
    #                              instant_death_p=.005)
    #
    #     print("done")
    #
    #     steps = agent.q_learning_experience(num_episodes=100,rate=rate,epsilon=1., batch_size=8, environment=game,watch=False,epsilon_factor=lambda x: 1./(1+x)**.5)
    #
    #     step_count_list.append(steps[0])
    #     print(step_count_list)
    #


    # game = gridworld.BoxGame(40,
    #     block_rate=.1,
    #     num_exits=2,
    #     num_pits=0,
    #     num_empty_targets=0,
    #     watch_construction=False,
    #     max_tree_steps=250,
    #     bias=.7,
    #     pad_size=agent_range)
    #
    # agent.set_environment(game)
    # agent.play(show_grid=True)
    #
    #
    #

