import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import qqdm
import os

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(2, -1),
        )

    def forward(self, x):
        return self.enc(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(1, 64, 2, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 128, 2, stride=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 256, 2, stride=2),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 3, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()  # 28*28*1
        )

    def forward(self, x):
        return self.dec(x.reshape(-1, 1, 4, 4))

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        code = self.encoder(x)
        return self.decoder(code)

    def save_encoder(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.encoder.state_dict(), "model/encoder_16.pt")

    def save_decoder(self):
        if not os.path.exists('model'):
            os.makedirs('model')
        torch.save(self.decoder.state_dict(), "model/decoder_16.pt")


if __name__ == '__main__':
    batch_size = 100
    train_datasets = torchvision.datasets.MNIST('./data', True, transform=torchvision.transforms.ToTensor(), download=True)
    train_dataloader = Data.DataLoader(dataset=train_datasets, shuffle=True, batch_size=batch_size)

    lr = 0.005
    epoch = 20
    autoencoder = Autoencoder()
    autoencoder.cuda()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    img_path = 'feature16/CAE_graphs'
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    for m in range(epoch):
        processbar = qqdm.qqdm(train_dataloader)
        step = 600 * m
        for i,(data, label) in enumerate(processbar):
            img = data.cuda()
            decoder_out = autoencoder(img)
            loss = loss_fn(decoder_out, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            processbar.set_infos({
                'loss': loss,
                'epoch': m,
                'step': step
            })
            step += 1
            if step % 600 ==0:
                tmp = decoder_out.cpu().reshape((batch_size, 1, 28, 28))
                tmp = torchvision.utils.make_grid(tmp, nrow=10)
                plt.figure(figsize=(10, 10))
                plt.imshow(tmp.permute(1, 2, 0))
                plt.savefig('{}/fake-{}.png'.format(img_path, step), format="PNG")
                plt.close()

    autoencoder.save_encoder()
    autoencoder.save_decoder()
    if not os.path.exists('result'):
        os.makedirs('result')
    torch.save(optimizer.state_dict(), "result/optimizer_16.pt")