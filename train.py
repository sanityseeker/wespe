import os
import random
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from matplotlib import pyplot as plt
import tqdm

from model import Generator, Discriminator, Vgg19
import dataset
from plots import TrainingPlotter, plot_iter
import blur
import saver

from warnings import filterwarnings

filterwarnings('ignore')


def compute_generator_loss(color_adversarial_decisions, gray_adversarial_decisions,
                           enhanced_patch, reconstructed_patch, vgg_model, is_true, device):

    # Color loss
    #     discr_color_loss = F.binary_cross_entropy(color_adversarial_decisions, is_true)
    # Compute generator loss only on generated images.
    #     is_generated = 1 - is_true

    criterion = nn.BCEWithLogitsLoss().to(device)
    generator_color_loss = criterion(color_adversarial_decisions,
                                     is_true)

    # Texture loss
#     discr_texture_loss = F.binary_cross_entropy(gray_adversarial_decisions, is_true)
    generator_texture_loss = criterion(gray_adversarial_decisions,
                                       is_true)

    norm_const = torch.prod(torch.IntTensor(list(enhanced_patch.size())).detach())

    # TV loss
    h_tv = F.mse_loss(enhanced_patch[:, :, 1:, :], enhanced_patch[:, :, :-1, :], reduction='sum')
    w_tv = F.mse_loss(enhanced_patch[:, :, :, 1:], enhanced_patch[:, :, :, :-1], reduction='sum')
    tv_loss = (h_tv + w_tv) / norm_const

#     criterion_content = nn.L1Loss().to(device)
    criterion_content = nn.MSELoss().to(device)

    # Content loss
    # content_loss = F.mse_loss(vgg_model(enhanced_patch).detach(),
    #                           vgg_model(reconstructed_patch), reduction='mean')

    content_loss = criterion_content(vgg_model(enhanced_patch), vgg_model(reconstructed_patch)) / norm_const

#     print(f'Color:{0.005 * discr_color_loss}\nTexture:{0.005 * generator_texture_loss}\nTV:{10 * tv_loss}\nContent:{content_loss}\n')

    # Total loss
    generator_loss = 0.005 * (generator_color_loss + generator_texture_loss) + 10 * tv_loss + content_loss
    return generator_loss


def compute_discriminator_losses(generated_blurred, real_blurred, color_discriminator,
                                 generated_grayed, real_grayed, texture_discriminator, is_true, device):

    is_false = 1 - is_true

    criterion = nn.BCEWithLogitsLoss().to(device)

    texture_decisions_real = texture_discriminator(real_grayed)
    color_decisions_real = color_discriminator(real_blurred)

    # color_decisions_fake = color_discriminator(generated_blurred.detach())
    color_decisions_fake = color_discriminator(generated_blurred)
    discr_color_loss = (criterion(color_decisions_fake, is_false)
                        + criterion(color_decisions_real, is_true))

#     print(color_decisions_fake, color_decisions_real, discr_color_loss)

    # texture_decisions_fake = texture_discriminator(generated_grayed.detach())
    texture_decisions_fake = texture_discriminator(generated_grayed)
    discr_texture_loss = (criterion(texture_decisions_fake, is_false)
                          + criterion(texture_decisions_real, is_true))

#     print(texture_decisions_fake, texture_decisions_real, discr_texture_loss)

    return discr_color_loss, discr_texture_loss


def train_epoch_mod(models, optimizers, data_generator, device, vgg_model, gaussian_kernel,
                    gen_ratio=1, disc_ratio=1, adaptive_train=True, epoch_num=0):
    enhancer, inv_enhancer, color_discriminator, texture_discriminator = map(lambda x: x[1], models)
    enhancer_opt, color_discriminator_opt, texture_discriminator_opt = map(lambda x: x[1], optimizers)
    train_losses = [[], [], []]
    for (_, optimizer) in optimizers:
        optimizer.zero_grad()

    tq = tqdm.tqdm(data_generator)

    for batch_num, (phone_batch, dslr_batch) in enumerate(tq):
        phone_batch, dslr_batch = phone_batch.to(device), dslr_batch.to(device)
        enhanced = enhancer(phone_batch.float())
        reconstructed = inv_enhancer(enhanced)

        is_true = torch.ones(phone_batch.shape[0], dtype=torch.float,
                             device=device) - torch.rand(1).to(device) * 0.1  # label smoothing

        is_true = 1 - is_true  # flip labels

        fake_blurred = blur.apply_kernel(enhanced, gaussian_kernel)
        real_blurred = blur.apply_kernel(dslr_batch, gaussian_kernel)

        # for param in color_discriminator.parameters():
        #     param.requires_grad = False
        # for param in texture_discriminator.parameters():
        #     param.requires_grad = False

        fake_grayed = dataset.to_grayscale(enhanced)
        real_grayed = dataset.to_grayscale(dslr_batch)

        texture_decisions_fake = texture_discriminator(fake_grayed)
        color_decisions_fake = color_discriminator(fake_blurred)

        if batch_num % (gen_ratio + disc_ratio) in list(range(gen_ratio)):
            enhancer_opt.zero_grad()
            generator_loss = compute_generator_loss(color_decisions_fake,
                                                    texture_decisions_fake,
                                                    enhanced, reconstructed, vgg_model, is_true, device)
            generator_loss.backward()
            enhancer_opt.step()
            train_losses[2].append(generator_loss.item())

        else:
            # for param in color_discriminator.parameters():
            #     param.requires_grad = True
            # for param in texture_discriminator.parameters():
            #     param.requires_grad = True

            color_discriminator_opt.zero_grad()
            texture_discriminator_opt.zero_grad()

            discr_color_loss, discr_texture_loss = compute_discriminator_losses(fake_blurred, real_blurred, color_discriminator,
                                                                                fake_grayed, real_grayed, texture_discriminator, is_true, device)

            discr_loss = (discr_color_loss + discr_texture_loss) / 2

            discr_loss.backward()

            color_discriminator_opt.step()
            texture_discriminator_opt.step()

            train_losses[0].append(discr_color_loss.item())
            train_losses[1].append(discr_texture_loss.item())

        if batch_num % 500 == 0:
            plot_iter(images=(dslr_batch.detach().cpu(), enhanced.detach().cpu()),
                      filename=f'{str(epoch_num).zfill(3)}{str(batch_num).zfill(6)}')

        if batch_num >= (gen_ratio + disc_ratio):
            tq.set_postfix_str(
                f'GL:{train_losses[2][-1]:.5f}, CDL:{train_losses[0][-1]:.3f}, TDL:{train_losses[1][-1]:.3f}')

        if batch_num >= 20 and adaptive_train:
            if (np.mean(train_losses[2][-5:]) > 0.015):
                disc_ratio = 0
                gen_ratio = 1
            elif ((np.mean(train_losses[0][-5:]) + np.mean(train_losses[1][-5:])) / 2 < 0.55):
                gen_ratio = 3
                disc_ratio = 1
            else:
                gen_ratio = 1
                disc_ratio = 3

    return train_losses, (dslr_batch.detach().cpu(), enhanced.detach().cpu())


def prepare_datasets(phone, batch_size, train_size=30000, val_ratio=0.1):
    file_names = os.listdir(dataset.PATHTOPHONETEMPLATE.format(phone))
    random.shuffle(file_names)

    num_val = int(val_ratio * len(file_names))
    num_train = min(len(file_names) - num_val, train_size)

    train_names = file_names[:num_train]
    val_names = file_names[-num_val:]

    train_set = dataset.TrainingDataset(phone, train_names)
    training_generator = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_set = dataset.TrainingDataset(phone, val_names)
    val_generator = data.DataLoader(val_set, batch_size=batch_size, drop_last=True)
    return training_generator, val_generator


def prepare_models(device, resume):
    models = (
        ('enhancer', Generator()),
        ('inv_enhancer', Generator()),
        ('color_discriminator', Discriminator()),
        ('texture_discriminator', Discriminator(channels=1)),
    )

    for (_, model) in models:
        model.to(device)

    optimizers = (
        ('enhancer_opt', torch.optim.Adam(itertools.chain(
            models[0][1].parameters(), models[1][1].parameters()),
         lr=1e-4)),
        ('color_discriminator_opt',
         torch.optim.Adam(models[2][1].parameters(), lr=1e-4, betas=(0.5, 0.999))),
        ('texture_discriminator_opt',
         torch.optim.Adam(models[3][1].parameters(), lr=1e-4, betas=(0.5, 0.999))),
    )

    if resume:
        saver.restore_state(models, optimizers, 'models')

    return models, optimizers


def train(phone, batch_size, n_epochs, save_rate=1, train_size=30000, resume=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    training_generator, _ = prepare_datasets(phone, batch_size, train_size)
    models, optimizers = prepare_models(device, resume)
    gaussian_kernel = blur.gaussian_kernel().to(device)
    train_log = [[], [], []]
    plotter = TrainingPlotter()

    os.makedirs('models/', exist_ok=True)

    vgg_model = Vgg19(device)
    vgg_model.to(device)

    for epoch in range(n_epochs):
        train_loss, images = train_epoch_mod(models, optimizers, training_generator,
                                             device, vgg_model, gaussian_kernel, 1, 2, True, epoch)
        for i in range(len(train_loss)):
            train_log[i].extend(train_loss[i])
        plotter.show(train_log, images)
        print("Mean discr color loss {}, last discr color loss {}".format(np.mean(train_log[0]), train_log[0][-1]))
        print("Mean discr texture loss {}, last discr texture loss {}".format(np.mean(train_log[1]), train_log[1][-1]))
        print("Mean gen loss {}, last gen loss {}".format(np.mean(train_log[2]), train_log[2][-1]))

        if epoch < 1000 or not epoch % save_rate:
            plotter.save()
        if not epoch % save_rate:
            saver.save_state(models, optimizers, 'models', epoch, device)


def main():
    train('iphone', batch_size=22, n_epochs=200, save_rate=100, train_size=100000, resume=False)


if __name__ == '__main__':
    main()
