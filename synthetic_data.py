import numpy as np
import torch
# pip install python-ternary
import ternary

EPS = 1e-7


def sample_from_simplex(n_classes, size=1):
    """
    Implements the `Kraemer algorithm <http://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf>`_
    for sampling uniformly at random from the unit simplex. This implementation is adapted from this
    `post <https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex>_`.

    Args:
        n_classes: integer, number of classes (dimensionality of the simplex)
        size: number of samples to return

    Returns: `np.ndarray` of shape `(size, n_classes,)` if `size>1`, or of shape `(n_classes,)` otherwise

    """
    if n_classes == 2:
        u = np.random.rand(size)
        u = np.vstack([1-u, u]).T
    else:
        u = np.random.rand(size, n_classes-1)
        u.sort(axis=-1)
        _0s = np.zeros(shape=(size, 1))
        _1s = np.ones(shape=(size, 1))
        a = np.hstack([_0s, u])
        b = np.hstack([u, _1s])
        u = b-a
    if size == 1:
        u = u.flatten()
    return u


def temp_scale(scores, temperature):
    logits = inv_softmax(scores)
    return logits / temperature


def inv_softmax(x, c=torch.log(torch.tensor(10))):
    return torch.log(x) + c


def sample_points(num_samples, num_classes, temp1, temp2):
    logits_temp1 = []
    samples = []
    labels = []
    for _ in range(num_samples):
        # Sample a point from the simplex
        sample = sample_from_simplex(num_classes, 1)
        # Temp scale it to simulate our setting
        logit = temp_scale(torch.tensor(sample), temp1).unsqueeze(0)
        scores = torch.softmax(logit, dim=1)[0]
        logits_temp1.append(logit.numpy())
        samples.append(scores.numpy())
        # Sample y according to that point
        labels.append(np.random.choice(np.arange(0, num_classes), p=scores.numpy()))

    logits_temp1 = torch.tensor(np.array(logits_temp1)).squeeze()
    pred_scores_temp1 = torch.tensor(np.array(samples)).type(torch.FloatTensor)
    targets = torch.tensor(np.array(labels).astype('int64'))

    # Second temp scaling
    logits_temp2 = temp_scale(pred_scores_temp1, temp2)
    pred_scores_temp2 = torch.softmax(logits_temp2, dim=1)

    # Scores cannot be exactly 0 or 1
    pred_scores_temp1 = torch.clamp(pred_scores_temp1, min=EPS, max=1 - EPS)
    pred_scores_temp2 = torch.clamp(pred_scores_temp2, min=EPS, max=1 - EPS)

    return pred_scores_temp1, pred_scores_temp2, logits_temp1, logits_temp2, targets


def plot_scatterplot(pred_scores):
    # Sample trajectory plot
    figure, tax = ternary.figure(scale=1.0)
    figure.set_size_inches(10, 8)

    tax.boundary()

    # Plot the data
    tax.scatter(pred_scores, marker='o', color="#3E4A89", label="")
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f", offset=0.02)

    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.show()
    ternary.plt.show()


if __name__ == '__main__':
    num_samples = 1000
    num_classes = 3
    temp1 = 0.9
    temp2 = 0.6
    pred_scores_temp1, pred_scores_temp2, _, _, _ = sample_points(num_samples, num_classes, temp1, temp2)
    # The plotting only works for three classes
    plot_scatterplot(pred_scores_temp1)
    plot_scatterplot(pred_scores_temp2)
