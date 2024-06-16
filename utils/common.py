import torch
import torchvision
import matplotlib.pyplot as plt
import os

def display_image(image, filename):
    with torch.no_grad():
        img = torchvision.utils.make_grid(image, nrow=5)
        plt.figure(figsize=(20, 20))
        plt.imshow(img.permute(1, 2, 0), cmap='Greys')
        plt.savefig('{}.png'.format(filename), format="PNG")
        plt.close()


def display_ECG(image, filename, dim):
    with torch.no_grad():
        image = image.reshape(-1, dim)
        plt.figure(figsize=(20, 20))
        num = len(image) if len(image) < 25 else 25
        for i in range(num):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.plot(image[i])
        plt.savefig('{}.png'.format(filename), format="PNG")
        plt.close()


def draw_line(x, y, labels, filename):
    fig, ax = plt.subplots()
    for x1, y1, lab in zip(x, y, labels):
        ax.plot(x1, y1, label=lab)
    ax.legend()
    plt.savefig('{}.png'.format(filename), format="PNG")
    plt.close()


def hellinger_distance(p, q, eps=1e-10):
    p = torch.sqrt(p + eps)
    q = torch.sqrt(q + eps)
    distance = torch.norm(p - q, dim=0)
    return distance


def init_noises(shape, mean=0, std=1):
    return torch.randn(shape) * std + mean