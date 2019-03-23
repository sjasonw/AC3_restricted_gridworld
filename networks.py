"""
 This is a module for artificial neural networks that are used for action-value
 functions, policy networks, etc.
"""

import torch
import torch.nn as nn
import tools


class ACNetDiscrete(nn.Module):
    def __init__(self, state_size, num_hidden=64, sigmoid_critic=True):
        super(ACNetDiscrete, self).__init__()

        self.conv_temp = nn.Sequential(
            # 2, 2, 11, 11
            nn.Conv3d(2, 20, kernel_size=(2, 4, 4)),
            nn.ReLU(),
            # 20, 1, 8, 8
            nn.MaxPool3d(kernel_size=(1, 2, 2))
            # 20, 1, 4, 4
        )

        self.conv_single = nn.Sequential(
            # 2, 11, 11
            nn.Conv2d(2, 10, kernel_size=3, stride=2),
            nn.ReLU(),
            # 10, 5, 5
            nn.Conv2d(10, 100, kernel_size=5)
            # 100, 1, 1
        )

        conv_temp_ft = 20*4*4
        single_frame_ft = 2*11*11
        conv_single_ft = 100


        fc1_in_ft = conv_temp_ft + single_frame_ft + conv_single_ft
        fc2_in_ft = num_hidden + state_size

        self.fc1 = nn.Sequential(
            nn.Linear(fc1_in_ft, num_hidden),
            nn.ReLU(),
            nn.Dropout(.3)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(fc2_in_ft, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.3)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )
        if not sigmoid_critic:
            self.critic = nn.Linear(num_hidden, 1)

        self.move_end = nn.Sequential(
            nn.Linear(num_hidden, 4),
            nn.LogSoftmax(dim=1)  # not the batch dimension!
        )

        self.internal_end = nn.Sequential(
            nn.Linear(num_hidden, 3),
            nn.LogSoftmax(dim=1)  # not the batch dimension!
        )

    def forward(self, frames, internal_state, mode):
        """

        :param frames:
            size (batch_size (optional), 2 (time), 2 (ch), 11, 11)
        :return:
        """
        if frames.dim() == 4:  # if there is no batch dimension
            frames = frames.unsqueeze(0)
        frames = frames.transpose(1, 2)  # make time a convolutional dimension
        assert frames.dim() == 5, "wrong frame dimension"
        if internal_state.dim() == 1:
            internal_state = internal_state.unsqueeze(0)
        assert internal_state.dim() == 2, "wrong internal state shape"

        last_frame = frames.select(dim=2, index=-1)
        last_frame_flat = last_frame.view(
            -1, tools.num_flat_features(last_frame))
        conv_temp_out = self.conv_temp(frames)
        conv_single_out = self.conv_single(last_frame)

        conv_temp_out = conv_temp_out.view(
            -1, tools.num_flat_features(conv_temp_out))
        conv_single_out = conv_single_out.view(
            -1, tools.num_flat_features(conv_single_out))
        fc1_in = (last_frame_flat, conv_single_out, conv_temp_out)
        fc1_in = torch.cat(fc1_in, dim=1)
        fc1_out = self.fc1(fc1_in)

        fc2_in = torch.cat((fc1_out, internal_state), dim=1)
        fc2_out = self.fc2(fc2_in)

        if mode == "critic":
            return self.critic(fc2_out)
        elif mode == "move":
            lp = self.move_end(fc2_out)
        elif mode == "internal":
            lp = self.internal_end(fc2_out)
        else:
            raise ValueError("mode not recognized")

        dist = torch.distributions.Categorical(torch.exp(lp))
        entropy = dist.entropy()
        selection = dist.sample()
        lp = [lp[i][selection[i]] for i in range(len(selection))]
        lp = torch.stack(lp)

        if mode == "internal":
            selection = selection.item() - 1   # 0,1,2 -> -1,0,1
        return selection, lp, entropy


class ActorCriticFrameMemory(nn.Module):

    def __init__(self, num_frames=40, num_hidden=100, sigmoid_critic=True):
        super(ActorCriticFrameMemory, self).__init__()
        assert num_frames == 40, "Currently num_frames must be 40."
        self.num_hidden = num_hidden
        self.num_protected_frames = 4

        self.frame_conv = nn.Sequential(
            nn.Conv3d(2, 10, kernel_size=(4, 4, 4), stride=(2, 1, 1))
        )

        self.frame_conv = nn.Sequential(
            # 40 11 11
            nn.Conv3d(2, 16, kernel_size=(5, 4, 4)),
            nn.ReLU(),
            # 36 8 8
            nn.MaxPool3d(kernel_size=(2, 1, 1)),
            nn.ReLU(),
            # 18 8 8
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3)),
            nn.ReLU(),
            # 14 6 6
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            nn.ReLU()
            # 7 3 3
        )
        processing_input_size = 7 * 3 * 3 * 32 + 2*11*11
        self.fc_processing = nn.Sequential(
            nn.Linear(processing_input_size, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.2)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )
        if not sigmoid_critic:
            self.critic = nn.Linear(num_hidden, 1)

        self.move_end = nn.Sequential(
            nn.Linear(num_hidden, 4),
            nn.LogSoftmax(dim=1)  # not the batch dimension!
        )

        self.delete_end = nn.Sequential(
            nn.Linear(num_hidden, num_frames-self.num_protected_frames),
            nn.LogSoftmax(dim=1)  # not the batch dimension!
        )

        discriminator_in_ft = 2 * 11 * 11 + processing_input_size
        self.discriminator_end = nn.Sequential(
            nn.Linear(discriminator_in_ft, 1),
            nn.Sigmoid()
        )

    def forward(self, frames: torch.Tensor, mode: str):

        if frames.dim() == 4:  # if there is no batch dimension
            frames = frames.unsqueeze(0)
        frames = frames.transpose(1, 2)  # make time a convolutional dimension
        assert frames.dim() == 5, "wrong frame dimension"
        last_frame = frames.select(dim=2, index=-1)
        last_frame = last_frame.view(-1, tools.num_flat_features(last_frame))
        conv_out = self.frame_conv(frames)
        conv_out = conv_out.view(-1, tools.num_flat_features(conv_out))
        proc_in = torch.cat((last_frame, conv_out), dim=1)
        fc_out = self.fc_processing(proc_in)

        if mode == "critic":
            return self.critic(fc_out)

        if mode == "move":
            lp = self.move_end(fc_out)

        elif mode == "delete":
            lp = self.delete_end(fc_out)

        else:
            raise ValueError("mode not recognized")

        dist = torch.distributions.Categorical(torch.exp(lp))
        entropy = dist.entropy()
        selection = dist.sample()
        lp = [lp[i][selection[i]] for i in range(len(selection))]
        lp = torch.stack(lp)
        return selection, lp, entropy

    def discriminate(self, frames, test_frame):
        if frames.dim() == 4:  # if there is no batch dimension
            frames = frames.unsqueeze(0)
        if test_frame.dim() == 3:
            test_frame = test_frame.unsqueeze(0)
        frames = frames.transpose(1, 2)  # make time a convolutional dimension
        assert frames.dim() == 5, "wrong frame dimension"

        last_frame = frames.select(dim=2, index=-1)
        last_frame = last_frame.view(-1, tools.num_flat_features(last_frame))
        conv_out = self.frame_conv(frames)
        conv_out = conv_out.view(-1, tools.num_flat_features(conv_out))
        proc_in = torch.cat((last_frame, conv_out), dim=1)
        fc_out = self.fc_processing(proc_in)

        discriminator_input = torch.cat((fc_out, test_frame), dim=1)
        return self.discriminator_end(discriminator_input)






#40 -c-> 36 -> 18 -c-> 12 -> ?
class ActorCriticNet1F(nn.Module):
    """
    Network input
    _____________
    frames : torch tensor
        size (temporal_dim, 11, 11)
    field : torch tensor
        size (field_ch, 28, 28)
    mode : str
        The value of mode must be either "actor" or "critic".  When
        mode is "actor" the return is a tuple (dist, log_pol_move)
        where dist is a torch.distribution.Normal object for
        writing to the field, and log_pol_move is the log of
        the probabilities of taking the four movement actions.
    """

    def __init__(self, field_ch, temporal_frames, num_hidden=100):
        super(ActorCriticNet1F, self).__init__()
        self.num_hidden = num_hidden

        field_ft = 20 * 4 * 4
        temp_ft = 10 * 4 * 4
        last_frame_ft = 2 * 11 * 11

        num_total_ft = field_ft + temp_ft + last_frame_ft

        # 28 x 28 -> 4 x 4
        self.field_conv = nn.Sequential(
            nn.Conv2d(field_ch, 10, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.temporal_conv = nn.Sequential(
            # (ch=2, num_frames, 11, 11) -> (ch=10, 1, 8, 8)
            nn.Conv3d(2, 10, kernel_size=(temporal_frames, 4, 4)),
            nn.ReLU(),
            # (ch=10, 1, 8, 8) -> (ch=10, 1 , 4, 4)
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.processing = nn.Sequential(
            nn.Linear(num_total_ft, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.3),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU()
        )

        self.critic = nn.Linear(num_hidden, 1)
        self.log_pol_move = nn.Sequential(
            nn.Linear(num_hidden, 4),
            nn.Dropout(), # TODO this is a test feature
            nn.LogSoftmax(dim=0)
        )



        self.mean_generator = nn.Sequential(
            # square image dimension: 1
            nn.ConvTranspose2d(num_hidden, 16,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(field_ch * 64),
            # square image dimension: 4
            nn.ConvTranspose2d(16, 8,
                               kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(field_ch * 32),
            # square image dimension: 10
            nn.ConvTranspose2d(8, 4,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(field_ch * 16),
            # square image dimension: 13
            nn.ConvTranspose2d(4, field_ch,
                               kernel_size=4, stride=2),
            # square image dimension: 28
            nn.Tanh()
        )

        self.sd_generator = nn.Sequential(
            # square image dimension: 1
            nn.ConvTranspose2d(num_hidden, 16,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(field_ch * 64),
            # square image dimension: 4
            nn.ConvTranspose2d(16, 8,
                               kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(field_ch * 32),
            # square image dimension: 10
            nn.ConvTranspose2d(8, 4,
                               kernel_size=4, stride=1, bias=False),
            nn.ReLU(),
            #nn.BatchNorm2d(field_ch * 16),
            # square image dimension: 13
            nn.ConvTranspose2d(4, field_ch,
                               kernel_size=4, stride=2),
            # square image dimension: 28
            nn.Sigmoid()
        )

        discriminator_in_ft = last_frame_ft + field_ft
        self.discriminator_end = nn.Sequential(
            nn.Linear(discriminator_in_ft, 1),
            nn.Sigmoid()
        )

    def forward(self, frames, field, mode: str):

        if mode == "test_memory":
            # In this mode, field must be a batch.
            field_ft = self.field_conv(field)
            field_ft = field_ft.view(-1, tools.num_flat_features(field_ft))
            frames = frames.view(-1, tools.num_flat_features(frames))
            discriminator_input = torch.cat((field_ft, frames), dim=1)
            return self.discriminator_end(discriminator_input)


        field_ft = self.field_conv(field.unsqueeze(0))
        field_ft = field_ft.flatten()

        reshaped = frames.transpose(0, 1).unsqueeze(0)
        temp_frame_ft = self.temporal_conv(reshaped)
        temp_frame_ft = temp_frame_ft.flatten()

        last_frame = frames[-1].flatten()

        processing_input = torch.cat((field_ft, temp_frame_ft, last_frame))
        out = self.processing(processing_input)

        if mode == "critic":
            return self.critic(out)
        elif mode == "move":
            log_pol_move = self.log_pol_move(out)
            return log_pol_move
        elif mode == "write":
            mean = self.mean_generator(out.view(1, self.num_hidden, 1, 1))
            sd = self.sd_generator(out.view(1, self.num_hidden, 1, 1))
            mean = mean.squeeze(dim=0)
            sd = sd.squeeze(dim=0)
            sd = sd * 1e-9  # TODO: this is a short-term fix attempt
            dist = torch.distributions.Normal(loc=mean, scale=sd)
            return dist
        else:
            raise ValueError("mode {0} not recognized".format(mode))





class ActorCriticNoField(nn.Module):
    """
    A version of the actor critic network with no field reading or writing.
    This is designed for testing AC3 code without the complication of
    the internal memory field.
    """

    def __init__(self, field_ch, temporal_frames, num_hidden=100):
        super(ActorCriticNoField, self).__init__()
        self.num_hidden = num_hidden
        temp_ft = 10 * 4 * 4
        last_frame_ft = 2 * 11 * 11

        num_total_ft = temp_ft + last_frame_ft


        self.temporal_conv = nn.Sequential(
            # (ch=2, num_frames, 11, 11) -> (ch=10, 1, 8, 8)
            nn.Conv3d(2, 10, kernel_size=(temporal_frames, 4, 4)),
            nn.ReLU(),
            # (ch=10, 1, 8, 8) -> (ch=10, 1 , 4, 4)
            nn.MaxPool3d(kernel_size=(1, 2, 2))
        )

        self.processing = nn.Sequential(
            nn.Linear(num_total_ft, num_hidden),
            nn.ReLU(),
            nn.Dropout(p=.3),
            nn.Linear(num_hidden, num_hidden),
            nn.ReLU()
        )

        self.critic = nn.Linear(num_hidden, 1)
        self.log_pol_move = nn.Sequential(
            nn.Linear(num_hidden, 4),
            nn.LogSoftmax(dim=0)
        )

    def forward(self, frames, mode: str):

        reshaped = frames.transpose(0, 1).unsqueeze(0)
        temp_frame_ft = self.temporal_conv(reshaped)
        temp_frame_ft = temp_frame_ft.flatten()

        last_frame = frames[-1].flatten()

        processing_input = torch.cat((temp_frame_ft, last_frame))
        out = self.processing(processing_input)

        if mode == "critic":
            return self.critic(out)
        elif mode == "move":
            log_pol_move = self.log_pol_move(out)
            return log_pol_move
        elif mode == "write":
            raise ValueError("write mode is not available for this network.")
        else:
            raise ValueError("mode {0} not recognized".format(mode))


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
            "open": 0.,
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
    ac = ACNetDiscrete(12)
    frames = torch.rand(2, 11, 11)
    frames = sens(frames)

    internal = torch.rand(12)
    out = ac(frames, internal, mode="internal")


    def print_num_params(model):

        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        print(params)

    print_num_params(ac)





