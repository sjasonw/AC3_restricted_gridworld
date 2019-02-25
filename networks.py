"""
 This is a module for artificial neural networks that are used for action-value
 functions and potentially other applications.
"""

import torch
import torch.nn as nn
import tools


# --------------------------------------------------------------------------
# Sensory network with fixed parameters
# --------------------------------------------------------------------------

class SensoryNetFixed(nn.Module):
    """
    This network, which has fixed predetermined weights and biases,
    is used to pre-processes agent views.  The original agent view
    is an (n,n) tensor for a small number n where each value is
    an integer representing the object located at the grid location.
    This network converts a batch of such images to a tensor of size
    (batch size, 2, n^2) before flattening to (batch size, 2*n^2)
    upon output.  The 2 should be understood as follows.  Consider
    batch number b and grid point (i,j).   The output with
    fixed (b, i, j) is

    out[b][:][i][j] =
        (1,1) if there is an exit at ij
        (0,1) if there is an open spot at ij
        (0,0) if there is a block at ij

    Note that the agent is not supposed to be able to see a "pit" so
    there is no need for another dimension here.
    """

    def __init__(self, rigidity=5.):
        super(SensoryNetFixed, self).__init__()
        self.convovle = nn.Conv2d(1, 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        for p in self.convovle.parameters():
            p.requires_grad = False
        key = {
            "exit": 2.,
            "open": 1.,
            #"block": -1.,
            # "pit": -2.
        }

        weights = rigidity * torch.ones((2,1,1,1))
        biases = rigidity*torch.tensor(
            [-(key["exit"] - .5), -(key["open"] - .5)]
        )
        self.convovle.weight.data = weights
        self.convovle.bias.data = biases

    def forward(self, x):
        x = x.unsqueeze(1)
        out = self.sigmoid(self.convovle(x))

        return out.view(-1, tools.num_flat_features(out))


class ActionValueNet(nn.Module):
    """
    This recurrent ANN is designed to approximate an action-value function
    where states are sequences of frames.  The input sequence should
    first be preprocessed by an object of class SensoryNetFixed.

    The last sequential element of the output of the final layer of the RNN
    is processed through a fully-connected layer with four output features
    representing the values of the four possible actions.

    For an agent to navigate, it is necessary to evaluate this network at
    each step.  Rather than re-inputting the entire episode frame sequence
    at each step, this network outputs the final sequential hidden
    tensors (h and c) which can then be re-entered into the network during
    the next run to start where you left off.  This could lead to some error
    if training loops are made in between time steps.
    """
    def __init__(self, agent_range, hidden_size=60, rnn_depth=2):
        super(ActionValueNet, self).__init__()
        self.width = 2*agent_range + 1
        self.rnn = nn.LSTM(
            input_size=2 * self.width**2,
            hidden_size=hidden_size,
            num_layers=rnn_depth,
            dropout=.4
        )
        #self.fc = nn.Linear(hidden_size, 4)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Dropout(p=.3),
            nn.Linear(hidden_size,4)
        )

    def forward(self,  sequence, h_and_c=None):
        """
        :param sequence:
        torch.Tensor with size (sequence len, batch, input size) where
        input size is the non-batch component of the output size of
        an object of class SensoryNetworkFixed.  This is probably equal to
        2 * self.width**2
        Alternatively, input can be a torch.nn.utils.rnn.PackedSequence.

        :param h_and_c:
        initialize the RNN's hidden variable with this input.

        :return:
        """
        # TODO delete me
        #print("value network called with  sequence ending in")
        #print("input seq size: ", sequence.data.size())



        if h_and_c is None:
            (outs, hidden) = self.rnn(sequence)
        else:
            #print("and hidden state")
            #print(h_and_c[0])
            (outs, hidden) = self.rnn(sequence, h_and_c)

        #print("hidden output:", hidden)

        if isinstance(sequence, nn.utils.rnn.PackedSequence):
            unpacked = nn.utils.rnn.pad_packed_sequence(outs)
            seqs = unpacked[0]
            seq_lens = unpacked[1]
            batch_size = len(seq_lens)
            last_outs = [seqs[seq_lens[i]-1][i] for i in range(batch_size)]
            last_outs = torch.stack(last_outs)

        elif isinstance(sequence, torch.Tensor):
            last_outs = outs[-1]

        values = self.fc(last_outs)

        return values, hidden

    def hidden_state_shape(self, num_batch=1):
        num_layers = self.rnn.num_layers
        num_hidden = self.rnn.hidden_size

        return torch.Size([num_layers, num_batch, num_hidden])

#
# # --------------------------------------------------------------------------
# # Sensory network input size (11,11)
# # --------------------------------------------------------------------------
#
# class SensoryNet11(nn.Module):
#     """
#     This is a hard-coded network in the sense that it is built only for
#     agents with a view range of exactly 5 so that the view size is (11,11).
#     For this project we are unlikely to focus on varying the agent's range
#     so there might not be much lost by restricting to this case only.
#     """
#
#     def __init__(self, agent_range):
#         super(SensoryNet11, self).__init__()
#         assert agent_range == 5, "For this network, the range must be 5."
#         self.range = 5
#         # the convolutional network sends a single channel with size
#         # (11, 11) to 30 channels with size (3,3)
#         self.conv_processing = nn.Sequential(
#             nn.Conv2d(1,15,kernel_size=3),
#             nn.BatchNorm2d(15),
#             nn.ReLU(),
#             nn.Conv2d(15,30,kernel_size=4),
#             nn.BatchNorm2d(30),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#
#         # "Whiskers" refer to the agent's input of the 8 nearest points on
#         # the grid (not including the agent's own location). We separately
#         # process this input so as to guarantee a high resolution perception
#         # of nearby objects.
#         self.whisker_processing = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=5, kernel_size=1),
#             nn.ReLU()
#         )
#
#     def forward(self, x):
#         """
#         :param x:
#         Batch of agent views size (batch size, 11, 11)
#         :return:
#         Output with size (batch size, 310)
#         """
#         whisker_input = self.extract_local_part(x)
#         # unsqueeze makes a single channel
#         conv_out = self.conv_processing(x.unsqueeze(1))
#         whisker_out = self.whisker_processing(whisker_input.unsqueeze(1))
#         flat_1 = conv_out.view(-1, tools.num_flat_features(conv_out))
#         flat_2 = whisker_out.view(-1, tools.num_flat_features(whisker_out))
#         return torch.cat((flat_1, flat_2), 1)
#
#     def extract_local_part(self, full_input):
#         """
#         :param full_input: input image seen by an agent
#         :return:
#         only the part of the image seen with range 1, flattened with
#         the central point removed.  These are the "whiskers"
#         """
#         a = self.range - 1
#         b = self.range + 2
#         local_input = full_input[:, a:b, a:b]
#         local_input = local_input.contiguous().view(-1, 9)
#         # remove central input
#         local_input = torch.cat(
#             (local_input[:, 0:4], local_input[:, 5:]),
#             dim=1
#         )
#         return local_input
#
#
#
#
#
#
# # --------------------------------------------------------------------------
# # Sensory network
# # --------------------------------------------------------------------------
#
# class SensoryNet(nn.Module):
#
#     def __init__(self, agent_range):
#
#         super(SensoryNet, self).__init__()
#
#         assert isinstance(agent_range, int) and agent_range > 0
#         self.range = agent_range
#         self.width = 2 * agent_range + 1
#
#         self.convolutional = nn.Sequential(
#             nn.Conv2d(in_channels=1,out_channels=5,kernel_size=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#
#         # "whiskers" refer to the agent's view of the 8 nearest points on
#         # the grid (not including the agent's own location). We separately
#         # process this input so as to guarantee a high resolution perception
#         # of nearby objects.
#         self.whisker_processing = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=5,kernel_size=1),
#             nn.ReLU()
#         )
#
#     def forward(self, agent_input):
#
#         # The "whiskers" see only close range input
#         whisker_input = self.extract_local_part(agent_input)
#
#         conv_out = self.convolutional(agent_input.unsqueeze(1))
#         #print(conv_out[-1].size())
#         import numpy as np
#         #import matplotlib.pyplot as plt
#         # if np.random.random() < .01:
#         #     print(agent_input[-1])
#         #     print(conv_out[-1][3])
#             #plt.close()
#             #tools.show_images([conv_out[-1][i].detach().numpy() for i in range(2)])
#         whisker_out = self.whisker_processing(whisker_input.unsqueeze(1))
#
#         flat_1 = conv_out.view(-1,tools.num_flat_features(conv_out))
#         flat_2 = whisker_out.view(-1, tools.num_flat_features(whisker_out))
#
#         return torch.cat((flat_1, flat_2), 1)
#
#     def extract_local_part(self, full_input):
#         """
#
#         :param full_input: input image seen by an agent
#         :return:
#         only the part of the image seen with range 1, flattened with
#         the central point removed.  These are the "whiskers"
#         """
#         a = self.range - 1
#         b = self.range + 2
#         local_input = full_input[:, a:b, a:b]
#         local_input = local_input.contiguous().view(-1, 9)
#
#         # remove central input
#         local_input = torch.cat(
#             (local_input[:,0:4], local_input[:,5:]),
#             dim=1
#         )
#
#         return local_input
#
#
# # --------------------------------------------------------------------------
# # Sequence processing network
# # --------------------------------------------------------------------------
#
# class ValueRNN(nn.Module):
#
#     def __init__(self, agent_range):
#
#         super(ValueRNN, self).__init__()
#
#         # There are 8 whiskers and 5 convolutional
#         # channels from the sensory layer
#         num_inputs = 5 * (agent_range**2 + 8)
#
#         self.rnn_1 = torch.nn.LSTM(
#             input_size=num_inputs,
#             hidden_size=100,
#             num_layers=1,
#             bias=True,
#             batch_first=False
#         )
#
#         self.drop = nn.Dropout(.3)
#
#         self.rnn_2 = torch.nn.LSTM(
#             input_size=100,
#             hidden_size=4,
#             num_layers=1,
#             bias=True,
#             batch_first=False
#         )
#
#     def forward(self, inp):
#
#         out = self.rnn_1(inp)[0]
#         #out = self.drop(out)
#         out = self.rnn_2(out)
#
#         #return out[0]
#         return out
#
# #
# #
# #
# #
# #
# #
# #
# #


if __name__ == "__main__":

    sensory = SensoryNetFixed()
    values = ActionValueNet(agent_range=5,hidden_size=7,rnn_depth=3)

    s0 = torch.ones(1,11,11)
    s1 = torch.rand(1, 11, 11)
    s_extra = torch.rand(1,11,11)

    r0 = torch.ones(1, 11, 11)
    r1 = torch.rand(1, 11, 11)
    r2 = torch.ones(1,11,11)
    r_extra = torch.ones(1,11,11)

    x0 = sensory(s0)
    x1 = sensory(s1)
    x_extra = sensory(s_extra)

    y0 = sensory(r0)
    y1 = sensory(r1)
    y2 = sensory(r2)
    y_extra = sensory(r_extra)

    x = torch.stack((x0, x1)).squeeze()
    y = torch.stack((y0,y1,y2)).squeeze()

    extras = torch.stack((y_extra, x_extra),dim=1)
    #extras.unsqueeze_(1) # make a trivial batch
    print("extras size:  ", extras.size())

    pack = nn.utils.rnn.pack_sequence([y,x])
    print("pack input size: ", pack.data.size())
    val, hid = values(pack)

    print(values(extras, hid)[0])

    full_x = torch.stack((x0, x1,x_extra)).squeeze()
    full_y = torch.stack((y0, y1, y2,y_extra)).squeeze()
    full_pck = nn.utils.rnn.pack_sequence([full_y, full_x])
    print(values(full_pck)[0])


















