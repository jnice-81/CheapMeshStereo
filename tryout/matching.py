import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torchvision import transforms
from PIL import Image
from typing import Tuple

class UNet(nn.Module):
    def __init__(self, levels):
        super(UNet, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()

        # Build encoder blocks
        in_channels = 1  # Assuming input has one channel, change accordingly
        for out_channels in levels:
            encoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.encoder_blocks.append(encoder_block)
            in_channels = out_channels

        # Build decoder blocks
        levels.reverse()
        for out_channels in levels[1:]:
            decoder_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(decoder_block)
            in_channels = out_channels

        # Last decoder block
        self.last_decoder_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),  # Assuming output has one channel, change accordingly
        )

    def forward(self, x):
        encoder_outputs = []

        # Encoder
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)

        # Decoder
        for block in self.decoder_blocks:
            x = block(x)

            # Concatenate with the corresponding encoder output
            encoder_output = encoder_outputs.pop()
            x = torch.cat([x, encoder_output], dim=1)

        # Last decoder block
        x = self.last_decoder_block(x)

        return x
    
class MatchNet(nn.Module):
    def __init__(self, pyramid_f_size) -> None:
        super(MatchNet, self).__init__()
        self.encoder = UNet(pyramid_f_size)

    def forward(self, imgL, imgR, max_disp):
        imgR : list= self.encoder(imgL)
        imgL : list= self.encoder(imgR)

        vals = torch.zeros((imgL.shape[0], max_disp, imgL.shape[2], imgL.shape[3]))
        for d in range(max_disp):
            until = imgL.shape[2] - d
            vals[:, d, :until, :] = torch.sum(
                imgL[:, :, :until, :] * imgR[:, :, d:, :],
                dim=1
            )
        
        return vals
    
class DrivingStereo(Dataset):
    def __init__(self, folder1_path, folder2_path, folder3_path):
        self.folder1_files = self.get_all_files(folder1_path)
        self.folder2_files = self.get_all_files(folder2_path)
        self.folder3_files = self.get_all_files(folder3_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def get_all_files(self, folder_path):
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)
        return all_files

    def __len__(self):
        return min(len(self.folder1_files), len(self.folder2_files), len(self.folder3_files))

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load and transform images from folder1 and folder2
        image1 = self.transform(Image.open(self.folder1_files[idx]))
        image2 = self.transform(Image.open(self.folder2_files[idx]))

        # Load and transform uint16 image from folder3
        image3 = Image.open(self.folder3_files[idx])
        image3 = transforms.ToTensor()(image3)
        image3 = image3 // 256

        return image1, image2, image3


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = MatchNet([3, 8, 16, 32, 64])
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

driving_stereo = DrivingStereo("/home/jannis/Downloads/Adirondack-perfect/lv", "/home/jannis/Downloads/Adirondack-perfect/rv", "/home/jannis/Downloads/Adirondack-perfect/disp")
train_size = int(1.0 * len(driving_stereo))
test_size = len(driving_stereo) - train_size
train_dataset, test_dataset = random_split(driving_stereo, [train_size, test_size], torch.Generator().manual_seed(0))
dataloaderTrain = DataLoader(train_dataset, batch_size=1, shuffle=True)
#dataloaderTest = DataLoader(test_dataset, batch_size=1, shuffle=True)


model.train()
for i in range(10):
    for j, (left, right, disp) in enumerate(dataloaderTrain):
        left.to(device)
        right.to(device)
        disp.to(device)

        optimizer.zero_grad()
        max_disp = torch.max(disp)
        print(max_disp)
        pred = model(left, right, max_disp)

        until = left.shape[2] - max_disp
        loss = criterion(pred[:, :, :(until), :], disp[:, :, :(until)].long())

        print(loss.item())

        loss.backward()
        optimizer.step()

        #if (i * len(train_dataset) + j) % 100 == 0:
            #with torch.no_grad():