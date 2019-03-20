import numpy as np
import matplotlib.pyplot as plt
import torch

def show_images(images, cols=1, titles=None):
    """
    Taken from soply on GitHub:
    https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

    Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in
                                 range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image,cmap='Greys', vmin=-2.,vmax=2.)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images/2)
    plt.show(block=False)
    plt.pause(.001)


def almost_same(x, y):
    """
    Check if two numbers that are already known to be within .1 of an integer
    are representing the same integer.

    Inputs can be floats, ints, mixed, etc.
    """
    return abs(x - y) < .2


def fair_argmax(x):
    """
    :param x: torch array
    :return:  argument with max value breaking ties evenly
    """
    assert x.dim() == 1, "input to fair_argmax must be 1 dimensional."
    max_indices = np.flatnonzero((x == x.max()).numpy())
    return np.random.choice(max_indices)


def perturb_blur(blur, pert_scale=.2):
    x0 = np.log(blur / (1 - blur))
    delta = pert_scale * np.random.randn()
    return sigmoid(x0 + delta)


def sigmoid(x):
    return 1. / (1.+ np.exp(-x))


def num_flat_features(state):
    size = state.size()[1:]
    flat_features = 1
    for k in size:
        flat_features *= k

    return flat_features

#
# def format_recall(recall):
#     """
#     TODO: add this documentation
#     """
#     batch_size = len(recall)
#     #print("formatting recall with lens ")
#     #print([len(recall[i]["frames"]) for i in range(batch_size)])
#     action_list = [recall[b]["action"] for b in range(batch_size)]
#     reward_list = [recall[b]["reward"] for b in range(batch_size)]
#     not_term_list = [not (recall[b]["terminal"]) for b in
#                      range(batch_size)]
#     h_list = [recall[b]["hidden"][0] for b in range(batch_size)]
#     c_list = [recall[b]["hidden"][1] for b in range(batch_size)]
#     back_frame_list = [recall[b]["frames"][:-1].squeeze(dim=1) for b in range(batch_size)]
#     fwd_frame_list = [recall[b]["frames"][-1] for b in range(batch_size)]
#
#     actions = torch.tensor(action_list)
#     rewards = torch.tensor(reward_list)
#     not_term = torch.tensor(not_term_list)
#     h = torch.cat(h_list, dim=1)  # batch_size is the second index
#     c = torch.cat(c_list, dim=1)  #
#     hidden = (h, c)
#     #print("back frame lens:  ", [len(back_frame_list[i]) for i in range(batch_size)])
#     back_frames = torch.nn.utils.rnn.pack_sequence(back_frame_list)
#     fwd_frames = torch.cat(fwd_frame_list)
#     output_dict = {
#         "hidden": hidden,
#         "back_frames": back_frames,
#         "forward_frames": fwd_frames,
#         "actions": actions,
#         "rewards": rewards,
#         "not_terminal": not_term
#     }
#     return output_dict

# def scored_permutation(score, x):
#     """
#     score and x are lists with the same length and score must have
#     ordered numerical value (int, float, etc).  Call score[i]
#     the "score of x[i]".  A permuted version of x is returned
#     with elements arranged so that scores go from highest to
#     lowest.
#     """
#
#     if isinstance(score, list):
#         score = torch.tensor(score)
#     (_, perm) = score.sort(descending=True)
#     return [x[perm[i]] for i in range(len(x))]
#
