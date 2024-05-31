import os

from matplotlib.figure import Figure

import constants


def save_plot_as_image(plot_figure: Figure, filename: str) -> None:
    plot_figure.savefig(os.path.join(constants.PLOT_DIR, filename))
