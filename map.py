import pennylane as qml
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from CAE_16 import Encoder, Decoder
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import qqdm
from utils.common import display_image

qubs = 4
dev = qml.device("default.qubit", wires=qubs)


def preprocess():
    data, labels = [], []
    for i, (d, label) in enumerate(dataloader):
        t = encoder(d).reshape(batch_size, -1)
        data.append(t.detach().numpy())
        labels.append(label.detach().numpy())

    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 16))
    sta_1 = np.sum(scaled_data ** 2, axis=1)
    sta_2 = np.sqrt(sta_1)
    return torch.from_numpy(scaled_data.reshape((-1, batch_size, 16))), torch.from_numpy(sta_2.reshape(-1, batch_size, 1))


@qml.qnode(dev, interface="torch")
def circuit(data):
    qml.AmplitudeEmbedding(data, normalize=True, wires=[0, 1, 2, 3])
    return qml.probs(wires=[0, 1, 2, 3])


class Mapper(nn.Module):
    def __init__(self):
        super(Mapper, self).__init__()
        self.mapper = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.ReLU(),
        )

    def save_model(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.state_dict(), "model/mapper.pt")

    def forward(self, x):
        return self.mapper(x)


def train():
    n, epoch, lr = 1, 20, 0.0001
    mapper = Mapper()
    optimizer = torch.optim.Adam(mapper.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    data, length = preprocess()

    step = 0
    for m in range(epoch):
        processbar = qqdm.qqdm(data)
        for i, dt in enumerate(processbar):
            deal = []
            for j, d in enumerate(dt):
                deal.append(circuit(d).detach().numpy())
            deal =torch.from_numpy(np.array(deal)).to(torch.float32)  # torch.Size([60, 16])
            out = mapper(deal)
            loss = loss_fn(out, length[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processbar.set_infos({
                'loss': loss,
                'epoch': m,
                'step': step
            })
            step += 1
            if step % 2000 == 0:
                new = torch.sqrt(deal) * out
                new = torch.from_numpy(scaler.inverse_transform(new.detach().numpy())).to(torch.float32)
                display_image(decoder(new).reshape(-1, 1, 28, 28), '{}/mapper-{}-{}'.format(img_path, n, step))

    mapper.save_model()
    if not os.path.exists('result'):
        os.makedirs('result')
    torch.save(optimizer.state_dict(), "result/optimizer_mapper.pt")


if __name__ == '__main__':
    batch_size = 25
    datasets = MNIST('./data', True, transform=torchvision.transforms.ToTensor(), download=True)
    dataloader = DataLoader(dataset=datasets, shuffle=True, batch_size=batch_size)
    scaler = MinMaxScaler(feature_range=(0, 1))

    device = torch.device('cpu')
    encoder = Encoder()
    if os.path.exists('model/encoder_16.pt'):
        encoder.load_state_dict(torch.load('model/encoder_16.pt', map_location=device))
    decoder = Decoder()
    if os.path.exists('model/decoder_16.pt'):
        decoder.load_state_dict(torch.load('model/decoder_16.pt', map_location=device))

    img_path = 'graphs_fake'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    train()