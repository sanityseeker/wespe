import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os

import matplotlib
matplotlib.use('Agg')


class TrainingPlotter(object):
    def __init__(self, figsize=(15, 10)):
        self.fig, self.ax = plt.subplots(4, 1, figsize=figsize)
        self.counter = 0

    def show(self, losses, images):
        discr_color_loss, discr_texture_loss, generator_loss = losses
        self.ax[0].clear()
        self.ax[0].plot(discr_color_loss, label='color loss', linestyle='--', linewidth=0.5)
        self.ax[0].plot(discr_texture_loss, label='texture loss', linestyle='--', linewidth=0.5)
        self.ax[0].plot((np.array(discr_color_loss) + np.array(discr_texture_loss)) / 2,
                        label='total loss', linewidth=2)
        self.ax[0].set_title('Discriminator', fontdict={'fontsize': 14, 'fontweight': 'medium'})
        self.ax[0].grid()
        self.ax[0].legend()
        self.ax[1].clear()
        self.ax[1].plot(generator_loss, label='generator loss')
        self.ax[1].grid()
        self.ax[1].legend()
        self.ax[1].set_title('Generator', fontdict={'fontsize': 14, 'fontweight': 'medium'})

        for i, image_batch in enumerate(images):
            grid = torchvision.utils.make_grid(image_batch[:8], nrow=len(image_batch[:8])).numpy()
            grid = grid.transpose((1, 2, 0))
            grid = np.clip(grid, 0, 1)
            self.ax[i + 2].imshow(grid)
            self.ax[i + 2].set_axis_off()
        self.fig.tight_layout()
        self.fig.show()
        plt.pause(0.01)

    def save(self):
        path = 'figs/epochs/'
        os.makedirs(path, exist_ok=True)
        self.fig.savefig(f'{path}{self.counter}.png')


def plot_iter(images, filename='image', figsize=(15, 10)):
    fig, ax = plt.subplots(2, 1, figsize=figsize)
    for i, image_batch in enumerate(images):
        grid = torchvision.utils.make_grid(image_batch[:8], nrow=len(image_batch[:8])).numpy()
        grid = grid.transpose((1, 2, 0))
        grid = np.clip(grid, 0, 1)
        ax[i].imshow(grid)
        ax[i].set_axis_off()

    fig.tight_layout()

    path = 'figs/iters/'
    os.makedirs(path, exist_ok=True)
    fig.savefig(f'{path}{filename}.png')
