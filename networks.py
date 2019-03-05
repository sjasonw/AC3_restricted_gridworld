"""
 This is a module for artificial neural networks that are used for action-value
 functions, policy networks, etc.
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
    (batch size, 2, n^2).  The 2 should be understood as follows.  Consider
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
        return self.sigmoid(self.convovle(x))


class FieldNet(nn.Module):
    """
    Our agent is often interested in processing 28 x 28 images with
    multiple channels which we call fields.  For example, the agent's
    "internal state" is a field used for memory.  It is useful to have
    one network for field processing to avoid redundant
    networks and parameters.  For example, the internal state
    must be used both to compute action values and to determine how
    to modify the internal state on a time step.  One object
    of class FieldNet can be called to perform both of these actions.

    This is a simple convolutional network.
    Input has size (batch size, channels, 28,28)
    and the output is a feature tensor with size
    (batch, channels * 40, 4, 4)

    TODO: Consider making the channels max out at 40 even for a multi-channel
    todo: field.  The growth of parameters may be too large to accommodate.
    """
    def __init__(self, channels):
        super(FieldNet, self).__init__()
        self.field_conv = nn.Sequential(
            # TODO: this is a potentially temporary modification:
            # ch x 28 x 28 --> 40 x 1 x 1
            nn.Conv2d(channels, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20,40,4),
            nn.ReLU()



            # # 28 x 28 -->  4 x 4
            # nn.Conv2d(channels * 1, channels * 20, 5),
            # nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(channels * 20, channels * 40, 5),
            # nn.ReLU(),
            # nn.MaxPool2d(2)



        )

    def forward(self, field):
        return self.field_conv(field)


class TemporalFrameNet(nn.Module):
    """

    """
    def __init__(self, agent_range, num_frames):
        super(TemporalFrameNet, self).__init__()
        # TODO: the agent range is required to be 5. Either make this a law or relax the requirement.
        assert agent_range == 5, "Range is locked to 5 for now."

        self.width = 2*agent_range + 1
        self.num_frames = num_frames
        self.temporal_conv = nn.Sequential(
            # ASSUMING agent_range == 5 !
            # (num_frames, 11, 11) -> (1, 8, 8)
            nn.Conv3d(2, 40, kernel_size=(num_frames, 4, 4)),
            nn.ReLU(),
            # (1, 8, 8) -> (1 , 4, 4)
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

    def forward(self, frames):
        """
        :param frames:
        Size (batch, num_frames, 2, 11, 11)
        2 comes from a SensoryNetFixed
        11 is because we have fixed agent range to 5 for now.

        :return:
        """
        # The temporal stack of frames is a dimension to convoluted
        return self.temporal_conv(frames.transpose(1,2))

class ConvolutionalReader(nn.Module):
    """
    This network applies a FieldNet and a TemporalFrameNet to an agent's
    internal (field) and external (frame) states.  The result is concatenated
    along with a direct copy of the last frame.  This tensor is then ready
    to be inputted either into a policy network or value network.
    """
    def __init__(self,field_channels, temporal_frames,ag_range):
        super(ConvolutionalReader, self).__init__()

        self.field_net = FieldNet(field_channels)
        # TODO: temp change
        #self.temporal_frame_net = TemporalFrameNet(ag_range, temporal_frames)
        # END

    def forward(self, field, frames):
        assert field.size()[0] == frames.size()[0], \
            "The internal field and frames must have the same batch size."
        assert field.dim() == 4, "invalid field size"
        assert frames.dim() == 5, "invalid frame size"
        # TODO: temp change
        #frames_tc = self.temporal_frame_net(frames)
        #frames_tc = frames_tc.view(-1, tools.num_flat_features(frames_tc))
        # END
        field = self.field_net(field)
        field = field.view(-1, tools.num_flat_features(field))
        last_frame = frames[:, -1]
        last_frame = last_frame.view(-1, tools.num_flat_features(last_frame))
        #TODO: temp
        #combined = torch.cat((field, last_frame, frames_tc), dim=1)
        combined = torch.cat((field, last_frame), dim=1)

        return combined


class ValueNet(nn.Module):
    """
    Overview:

    This network furnishes a state-action value function.

    # TODO: commentate

    The other input, frames, refers to the external agent view.
    This view must have already been processed by an object of class
    SensoryNetFixed.  However, unlike the field input, it is not enough
    to just hand a ValueNet the output of a SensoryNetFixed.  The reason
    is that this ValueNet is designed to look at a short sequence of
    outputs from SensoryNetFixed.  The output of SensoryNetFixed has
    size (batch, 2, width, width) where width refers to the width of
    the agent's viewable region.  Form a stack of num_frames of these
    output tensors with shape (batch, num_frames, 2, width, width) where
    the stack refers to the most recent num_frames frames that the agent
    has seen.  The order should be chronological in the second index.
    """
    def __init__(self, agent_range, field_ch, num_hidden=300):
        super(ValueNet, self).__init__()
        # TODO: the agent range is required to be 5. Either make this a law or relax the requirement.
        assert agent_range == 5, "Range is currently locked to 5."
        width = 2 * agent_range + 1

        # TODO temp
        #frame_conv_ft = 40 * 4 * 4
        #END

        last_frame_ft = 2 * width**2

        # TODO: this is a potential change
        # field_ft = field_ch * 40 * 4 * 4
        field_ft = 40
        #
        # TODO TEMP
        #linear_inputs = frame_conv_ft + last_frame_ft + field_ft
        #
        linear_inputs = last_frame_ft + field_ft

        self.final_processing = nn.Sequential(
            nn.Linear(linear_inputs, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.3),
            #TODO potential change
            #nn.Linear(num_hidden, 4)
            nn.Linear(num_hidden, 50),
            nn.ReLU(),
            nn.Linear(50,4)
            # END

        )

    def forward(self, state):
        """

        :return:
        torch.tensor size (batch, 4)
        Returns value of each of the agent's four possible walking actions.
        """
        return self.final_processing(state)


class InternalPolicy(nn.Module):
    """
    Overview:
    This network takes as input the agent's internal and external state.
    The output is a probability distribution.  The distribution can be sampled
    to select an action or its method log_prob can be called to obtain an
    eligibility vector.
    """
    def __init__(self, agent_range, field_ch, ft_multiplier=4):
        super(InternalPolicy, self).__init__()
        assert ft_multiplier >= field_ch


        width = 2*agent_range + 1
        self.ft_multiplier = ft_multiplier
        # TODO temp
        # frame_conv_ft = 40 * 4 * 4
        # end
        last_frame_ft = 2 * width ** 2
        # TODO: potential change
        #field_ft = field_ch * 40 * 4 * 4
        field_ft = 40
        #
        # TODO temp
        #self.total_features = frame_conv_ft + last_frame_ft + field_ft
        self.total_features = last_frame_ft + field_ft

        self.initial_processing = nn.Sequential(
            nn.Linear(self.total_features, 128),
            nn.ReLU(),
            nn.Dropout(p=.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=.2),
            nn.Linear(64, ft_multiplier * 4)
        )

        self.mean_generator = nn.Sequential(
            # square image dimension: 1
            nn.ConvTranspose2d(ft_multiplier * 4, ft_multiplier * 4,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ft_multiplier * 4),
            # square image dimension: 4
            nn.ConvTranspose2d(ft_multiplier * 4, ft_multiplier * 2,
                               kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ft_multiplier * 2),
            # square image dimension: 10
            nn.ConvTranspose2d(ft_multiplier * 2, ft_multiplier,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ft_multiplier),
            # square image dimension: 13
            nn.ConvTranspose2d(ft_multiplier, field_ch,
                               kernel_size=4, stride=2, bias=False),
            # square image dimension: 28
            nn.Tanh()
        )

        self.sd_generator = nn.Sequential(
            # square image dimension: 1
            nn.ConvTranspose2d(ft_multiplier * 4, ft_multiplier * 4,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ft_multiplier * 4),
            # square image dimension: 4
            nn.ConvTranspose2d(ft_multiplier * 4, ft_multiplier * 2,
                               kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ft_multiplier * 2),
            # square image dimension: 10
            nn.ConvTranspose2d(ft_multiplier * 2, ft_multiplier,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ft_multiplier),
            # square image dimension: 13
            nn.ConvTranspose2d(ft_multiplier, field_ch,
                               kernel_size=4, stride=2, bias=False),
            # square image dimension: 28
            nn.Sigmoid()
        )

    def forward(self, state):
        """

        :param state:
        torch.tensor size (batch, num_features)
        This is the output of an object of class ConvolutionalReader.

        :return:
        torch.distributions.Normal object with
        """
        assert state.size()[1] == self.total_features
        num_batches = state.size()[0]

        processed = self.initial_processing(state)

        processed = processed.view(-1, self.ft_multiplier*4, 1, 1)
        mean = self.mean_generator(processed)
        sd = self.sd_generator(processed)
        return torch.distributions.Normal(loc=mean, scale=sd)











#
#
#
# class ActionValueNet(nn.Module):
#     """
#     This recurrent ANN is designed to approximate an action-value function
#     where states are sequences of frames.  The input sequence should
#     first be preprocessed by an object of class SensoryNetFixed.
#
#     The last sequential element of the output of the final layer of the RNN
#     is processed through a fully-connected layer with four output features
#     representing the values of the four possible actions.
#
#     For an agent to navigate, it is necessary to evaluate this network at
#     each step.  Rather than re-inputting the entire episode frame sequence
#     at each step, this network outputs the final sequential hidden
#     tensors (h and c) which can then be re-entered into the network during
#     the next run to start where you left off.  This could lead to some error
#     if training loops are made in between time steps.
#     """
#     def __init__(self, agent_range, hidden_size=60, rnn_depth=2):
#         super(ActionValueNet, self).__init__()
#         self.width = 2*agent_range + 1
#         self.rnn = nn.LSTM(
#             input_size=2 * self.width**2,
#             hidden_size=hidden_size,
#             num_layers=rnn_depth,
#             dropout=.4
#         )
#         #self.fc = nn.Linear(hidden_size, 4)
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size,hidden_size),
#             nn.ReLU(),
#             nn.Dropout(p=.3),
#             nn.Linear(hidden_size,4)
#         )
#
#     def forward(self,  sequence, h_and_c=None):
#         """
#         :param sequence:
#         torch.Tensor with size (sequence len, batch, input size) where
#         input size is the non-batch component of the output size of
#         an object of class SensoryNetworkFixed.  This is probably equal to
#         2 * self.width**2
#         Alternatively, input can be a torch.nn.utils.rnn.PackedSequence.
#
#         :param h_and_c:
#         initialize the RNN's hidden variable with this input.
#
#         :return:
#         """
#         # TODO delete me
#         #print("value network called with  sequence ending in")
#         #print("input seq size: ", sequence.data.size())
#
#
#
#         if h_and_c is None:
#             (outs, hidden) = self.rnn(sequence)
#         else:
#             #print("and hidden state")
#             #print(h_and_c[0])
#             (outs, hidden) = self.rnn(sequence, h_and_c)
#
#         #print("hidden output:", hidden)
#
#         if isinstance(sequence, nn.utils.rnn.PackedSequence):
#             unpacked = nn.utils.rnn.pad_packed_sequence(outs)
#             seqs = unpacked[0]
#             seq_lens = unpacked[1]
#             batch_size = len(seq_lens)
#             last_outs = [seqs[seq_lens[i]-1][i] for i in range(batch_size)]
#             last_outs = torch.stack(last_outs)
#
#         elif isinstance(sequence, torch.Tensor):
#             last_outs = outs[-1]
#
#         values = self.fc(last_outs)
#
#         return values, hidden
#
#     def hidden_state_shape(self, num_batch=1):
#         num_layers = self.rnn.num_layers
#         num_hidden = self.rnn.hidden_size
#
#         return torch.Size([num_layers, num_batch, num_hidden])
#
#




if __name__ == '__main__':
    import numpy as np
    from time import time
    sens = SensoryNetFixed()
    reader = ConvolutionalReader(field_channels=2,temporal_frames=4,ag_range=5)
    val = ValueNet(agent_range=5,field_ch=2,num_hidden=40)

    v = torch.ones(11, 11)
    v2 = -1*torch.ones(11, 11)
    v3 = 0*torch.ones(11, 11)
    v4 = v

    frames = torch.stack((v,v2,v3, v4))
    frames = sens(frames)
    frames = torch.stack((frames, frames))  # batch size = 2
    print("frame size: ", frames.size())

    internal = torch.rand(2,2,28,28)
    t1 = time()
    combined = reader(internal, frames)
    out = val(combined)


    def print_num_params(model):

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print(params)

    print_num_params(val)

    print(out)

    policy = InternalPolicy(5,field_ch=2)
    policy(combined)




