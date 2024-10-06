import matplotlib.pyplot as plt


def plot_training(loss_func, title_name=None, file_name=None, fs=11):
    fig, ax = plt.subplots()
    ax.plot(loss_func, c="#f03b20")

    if title_name:
        plt.title(title_name, fontsize=fs)

    if file_name:
        fig.savefig(file_name + "_loss.png", format="png")

    return fig, ax


def plot_poincare_disk(points, filename="poincare_disk.png", titletext=""):
    """
    Plot points in the Poincaré disk and save the plot to a file.
    """
    fig, ax = plt.subplots()
    ax.plot(points[:, 0], points[:, 1], "o")
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    circle = plt.Circle((0, 0), 1, edgecolor="black", facecolor="none")
    ax.plot(0, ".", c="black")
    ax.add_artist(circle)
    return fig, ax


def plot_vector_field(points, titletext=""):
    """
    Plot points with vectors in phase-space of 2d base manifold
    """
    # split points into base point and tangent vector
    n = int(points.size(1) / 2)
    x = points[:, :n]
    v = points[:, n:]

    # set up background
    fig, ax = plt.subplots()
    ax.set_aspect("equal", "box")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    circle = plt.Circle((0, 0), 1, edgecolor="black", facecolor="none")
    ax.add_artist(circle)
    # plot base points
    ax.scatter(x[:, 0], x[:, 1])
    # add tangent vectors
    ax.quiver(x[:, 0], x[:, 1], v[:, 0], v[:, 1])
    return fig, ax


def plot_phase_space(x, v, titletext=""):
    """
    Plot projection of points in phase-space (tangent bundle) as scatter plot in 2d
    """
    fig, ax = plt.subplots()
    ax.scatter(x, v)
    return fig, ax
