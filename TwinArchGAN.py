import pennylane as qml
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from CAE_16 import Encoder, Decoder
from map import Mapper
from discriminator import Disc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from utils.common import *


qubs = 4
f_len = 2 ** qubs
dev = qml.device("default.qubit", wires=2 * qubs)
batch_size = 1
total = 300
n = 1
labels = [0]
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = MNIST('./data', True, transform=torchvision.transforms.ToTensor(), download=True)
datas = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size)

device = torch.device('cpu')
encoder = Encoder()
if os.path.exists('model/encoder_16.pt'):
    encoder.load_state_dict(torch.load('model/encoder_16.pt', map_location=device))
decoder = Decoder()
if os.path.exists('model/decoder_16.pt'):
    decoder.load_state_dict(torch.load('model/decoder_16.pt', map_location=device))
mapper = Mapper()
if os.path.exists('model/mapper.pt'):
    mapper.load_state_dict(torch.load('model/mapper.pt'))
classifier = Disc()
if os.path.exists('model/disc.pt'):
    classifier.load_state_dict(torch.load('model/disc.pt', map_location=device))


def preprocess():
    data = []
    for i, (d, label) in enumerate(datas):
        if label in labels:
            t = encoder(d).reshape(batch_size, -1)
            data.append(t.detach().numpy())
        if len(data) == total:
            break
    scaled_data = scaler.fit_transform(np.array(data).reshape(-1, f_len))
    return torch.from_numpy(scaled_data.reshape(-1, f_len))


@qml.qnode(dev, interface="torch")
def disc_circuit(inputs, params):
    for i in range(qubs):
        qml.RY(inputs[i], wires=i)
    for i in range(qubs):
        qml.RX(inputs[i + qubs], wires=i)
    qml.StronglyEntanglingLayers(weights=params, wires=[i for i in range(qubs)])
    return qml.probs(wires=[i for i in range(qubs)])


@qml.qnode(dev, interface="torch")
def disc_circuit1(inputs, params):
    qml.StronglyEntanglingLayers(weights=params, wires=[i for i in range(qubs)])
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(qubs)]


@qml.qnode(dev, interface="torch")
def gen_circuit(inputs, params):
    for i in range(qubs):
        qml.RY(inputs[i], wires=qubs+i)
    for i in range(qubs):
        qml.RX(inputs[i+qubs], wires=qubs + i)
    qml.StronglyEntanglingLayers(weights=params, wires=[qubs+i for i in range(qubs)])
    return qml.probs(wires=[qubs+i for i in range(qubs)])


@qml.qnode(dev, interface="torch")
def gen_circuit1(inputs, params):
    for i in range(qubs):
        qml.RY(inputs[i], wires=qubs+i)
    qml.StronglyEntanglingLayers(weights=params, wires=[qubs+i for i in range(qubs)])
    return [qml.expval(qml.PauliZ(wires=i + qubs)) for i in range(qubs)]


@qml.qnode(dev, interface="torch")
def data_circuit(inputs):
    qml.AmplitudeEmbedding(features=inputs, normalize=True, wires=[qubs + i for i in range(qubs)])
    return qml.probs(wires=[qubs+i for i in range(qubs)])


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=qubs)
        self.qlayer = qml.qnn.TorchLayer(gen_circuit, {'params': shape})
        self.qlayer1 = qml.qnn.TorchLayer(gen_circuit1, {'params': shape})
        self.qlayer2 = qml.qnn.TorchLayer(gen_circuit1, {'params': shape})
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x0 = self.qlayer1(x[:qubs])
        x1 = self.qlayer2(x[qubs:qubs*2])
        x4 = self.dropout(torch.cat((x0, x1), dim=0))
        out = self.qlayer(x4)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        shape = qml.StronglyEntanglingLayers.shape(n_layers=4, n_wires=qubs)
        self.qlayer = qml.qnn.TorchLayer(disc_circuit, {'params': shape})
        self.qlayer1 = qml.qnn.TorchLayer(disc_circuit1, {'params': shape})
        self.qlayer2 = qml.qnn.TorchLayer(disc_circuit1, {'params': shape})

    def forward(self):
        x0 = self.qlayer1(torch.tensor([]))
        x1 = self.qlayer2(torch.tensor([]))
        out = self.qlayer(torch.cat((x0, x1)))
        return out


class DataLoader(nn.Module):
    def __init__(self):
        super(DataLoader, self).__init__()
        self.loader = qml.qnn.TorchLayer(data_circuit, {})

    def forward(self, x):
        output = self.loader(x)
        return output


def generate_image(idx, num=200, filename='fake'):
    global test_disc, real_center
    with torch.no_grad():
        images, samples, feature = [], [], []
        for i in range(num):
            z = init_noises(qubs * 3, mean=0, std=3)
            t = generator(z)
            feature.append(t)
            t = t.reshape(1, -1).to(torch.float32)
            length = mapper(t).item()
            t = torch.sqrt(t) * length
            samples.append(t)
            t = torch.from_numpy(scaler.inverse_transform(t)).to(torch.float32)
            images.append(decoder(t).reshape(1, 28, 28))
        display_image(images[:25], '{}/{}-{}-{}'.format(img_path, filename, n, idx))


        if idx % 3 == 0 or idx == epoch - 1:
            images = torch.from_numpy(np.array(images))
            mean = torch.mean(images, dim=0)
            dis = torch.sum(torch.sqrt(torch.sum(torch.square(images-mean), dim=[-1, -2]))) / num
            distance.append(dis.item())

            x = torch.sqrt(real_center)
            fid = 0
            for y in feature:
                y = torch.sqrt(y)
                dot = torch.dot(x, y) ** 2
                fid += dot
            fidelity.append(fid.item() / num)

            p = torch.from_numpy(np.array(samples).reshape(-1, f_len)).softmax(dim=0)
            q = real_data.reshape(-1, f_len)
            q = q[:num].softmax(dim=0)
            hd = torch.sum(hellinger_distance(p, q)) / f_len
            hellinger_value.append(hd.item())

            outputs = classifier(images)
            _, predicted = torch.max(outputs.data, 1)
            total = len(images)
            correct = len([d for d in predicted if d in labels])
            validity.append(correct / total)
            # f.write('epoch: {}\n'.format(idx))
            # f.write(str(predicted.reshape(-1, 5)))
            subscript.append(idx)


def similarity(real, data, noise):
    global test_disc
    if real == 0:
        x = torch.sqrt(dataloader(data))
        y = torch.sqrt(discriminator())
        dot = torch.dot(x, y)
        norm_x = torch.norm(x)
        norm_y = torch.norm(y)
        return dot / (norm_x * norm_y)
    elif real == 1:
        x = torch.sqrt(generator(noise))
        y = torch.sqrt(discriminator())
        dot = torch.dot(x, y)
        norm_x = torch.norm(x)
        norm_y = torch.norm(y)
        return dot / (norm_x * norm_y)
    elif real == 2:
        x = torch.sqrt(generator(noise))
        y = torch.sqrt(discriminator())
        dot = torch.dot(x, y)
        norm_x = torch.norm(x)
        norm_y = torch.norm(y)
        return dot / (norm_x * norm_y)


# 损失函数
def gen_loss(e):
    print('gen:', e)
    return -torch.log(e)


def disc_loss(e, real):
    if real:
        print('disc_real: ', e)
        return -torch.log(e)
    else:
        print('disc_fake: ', e)
        return -torch.log(1-e)


# training
epoch = 100
d_nums = 60
g_nums = 80
real_data = preprocess()  # torch.Tensor
real_tmp = torch.from_numpy(scaler.inverse_transform(real_data[:200])).to(torch.float32)
real_img = decoder(real_tmp).reshape(200, -1)

discriminator = Discriminator()
generator = Generator()
dataloader = DataLoader()

opt1 = torch.optim.Adam(discriminator.parameters(), lr=0.001)
opt2 = torch.optim.SGD(generator.parameters(), lr=0.005)

test_disc = []
disc_fake = []
disc_real = []
real_center = None

real_variance = []
distance = []
validity = []
fidelity = []
subscript = []
hellinger_value = []
hellinger_value1 = []


def train():
    global real_center
    for i in range(epoch):
        z = init_noises(qubs * 3, mean=0, std=3)
        # train the discriminator on the real data
        for j, data in enumerate(real_data):
            loss = disc_loss(similarity(0, data, None), True)
            opt1.zero_grad()
            loss.backward()
            opt1.step()
        real_center = discriminator()

        # with torch.no_grad():
        #     t = discriminator().reshape(1, -1).to(torch.float32)
        #     length = mapper(t).item()
        #     t = torch.sqrt(t) * length
        #     t = torch.from_numpy(scaler.inverse_transform(t)).to(torch.float32)
        #     disc_real.append(decoder(t).reshape(1, 28, 28))

        # train the discriminator on the generator
        generator.eval()
        for j in range(d_nums):
            loss = disc_loss(similarity(1, None, z), False)
            opt1.zero_grad()
            loss.backward()
            opt1.step()
        generator.train()

        # with torch.no_grad():
        #     t = discriminator().reshape(1, -1).to(torch.float32)
        #     length = mapper(t).item()
        #     t = torch.sqrt(t) * length
        #     test_disc.append(t.reshape(-1))
        #     t = torch.from_numpy(scaler.inverse_transform(t)).to(torch.float32)
        #     disc_fake.append(decoder(t).reshape(1, 28, 28))

        # train the generator
        for j in range(g_nums):
            loss = gen_loss(similarity(2, None, z))
            opt2.zero_grad()
            loss.backward()
            opt2.step()

        generate_image(i)
        print('epoch: ', i)

    f.write('gen_params: {}\n'.format(list(generator.parameters())))
    # f.write('disc_params: {}\n'.format(disc_params))

    # 画图
    draw_line([subscript], [distance], ['distance'], '{}/{}-{}'.format(result_path, 'distance', n))
    df1 = pd.DataFrame({'index': subscript, 'distance': distance})
    draw_line([subscript], [validity], ['validity'], '{}/{}-{}'.format(result_path, 'validity', n))
    df2 = pd.DataFrame({'index': subscript, 'validity': validity})
    draw_line([subscript], [fidelity], ['fidelity'], '{}/{}-{}'.format(result_path, 'fidelity', n))
    df3 = pd.DataFrame({'index': subscript, 'fidelity': fidelity})
    draw_line([subscript], [hellinger_value], ['hellinger_value'], '{}/{}-{}'.format(result_path, 'hellinger_value', n))
    df4 = pd.DataFrame({'index': subscript, 'hellinger_value': hellinger_value})
    draw_line([subscript], [hellinger_value1], ['hellinger_value1'], '{}/{}-{}'.format(result_path, 'hellinger_value1', n))
    df5 = pd.DataFrame({'index': subscript, 'hellinger_value1': hellinger_value1})
    with pd.ExcelWriter('{}/{}.xlsx'.format(result_path, n)) as writer:
        df1.to_excel(writer, sheet_name='Sheet1', index=False)
        df2.to_excel(writer, sheet_name='Sheet2', index=False)
        df3.to_excel(writer, sheet_name='Sheet3', index=False)
        df4.to_excel(writer, sheet_name='Sheet4', index=False)
        df5.to_excel(writer, sheet_name='Sheet5', index=False)

    # display_image(disc_fake, '{}/disc_fake-{}'.format(img_path, n))
    # display_image(disc_real, '{}/disc_real-{}'.format(img_path, n))

if __name__ == '__main__':
    img_path = 'feature16/fake_img'
    result_path = 'feature16/result'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    f = open('{}/loss-{}.txt'.format(result_path, n), 'w')
    train()