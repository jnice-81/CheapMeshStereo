import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torchvision import transforms
from PIL import Image
from typing import Tuple
import torchvision.transforms.functional as TF
import pickle
import tqdm

class UNet(nn.Module):
    def __init__(self, levels):
        super(UNet, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.encoder_pools = nn.ModuleList()
        self.decoder_pools = nn.ModuleList()

        # Build encoder blocks
        in_channels = 3  # Assuming input has one channel, change accordingly
        for out_channels in levels:
            encoder_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.encoder_blocks.append(encoder_block)
            self.encoder_pools.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            in_channels = out_channels

        # Build decoder blocks
        levels.reverse()
        for out_channels in levels:
            decoder_block = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.decoder_pools.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            self.decoder_blocks.append(decoder_block)
            in_channels = out_channels

        # Last decoder block
        self.last_decoder_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        encoder_outputs = []
        encoder_indizes = []

        # Encoder
        for block, pool in zip(self.encoder_blocks, self.encoder_pools):
            x = block(x)
            encoder_outputs.append(x)
            x, ind = pool(x)
            encoder_indizes.append(ind)

        # Decoder
        for block, pool in zip(self.decoder_blocks, self.decoder_pools):
            x = block(x)
            ind = encoder_indizes.pop()
            encoder_output = encoder_outputs.pop()
            x = pool(x, ind, output_size=encoder_output.shape)
            
            # Concatenate with the corresponding encoder output
            x = x + encoder_output

        # Last decoder block
        x = self.last_decoder_block(x)

        return x
    
class MatchNet(nn.Module):
    def __init__(self, pyramid_f_size) -> None:
        super(MatchNet, self).__init__()
        self.encoder = UNet(pyramid_f_size)

    def forward(self, imgL, imgR, max_disp):
        imgR : torch.Tensor = self.encoder(imgR)
        imgL : torch.Tensor = self.encoder(imgL)

        vals = torch.zeros((imgL.shape[0], max_disp, imgL.shape[2], imgL.shape[3]), device=imgL.device)
        for d in range(max_disp):
            until = imgL.shape[3] - d
            vals[:, d, :, :until] = torch.sum(
                imgL[:, :, :, d:] * imgR[:, :, :, :until],
                dim=1
            )
        
        
        return vals

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

    
class DrivingStereo(Dataset):
    def __init__(self, folder_left, folder_right, folder_disp):
        self.folder_left = self.get_all_files(folder_left, "./imnamesl.pkl")    
        self.folder_right = self.get_all_files(folder_right, "./imnamesr.pkl")
        self.folder_disp = self.get_all_files(folder_disp, "./imnamesd.pkl")
        self.idx_to_val = {}
        for i, key in enumerate(self.folder_left):
            self.idx_to_val[i] = key
        self.transform = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def get_all_files(self, folder_path, cache_at):
        all_files = {}
        if os.path.exists(cache_at):
            with open(cache_at, "rb") as f:
                all_files = pickle.load(f)
        else:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files[file.split(".")[0]] = file_path
            with open(cache_at, "wb") as f:
                pickle.dump(all_files, f)
        
        return all_files

    def __len__(self):
        return min(len(self.folder_left), len(self.folder_right), len(self.folder_disp))

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vl = self.idx_to_val[idx]
        
        # Load and transform images from folder1 and folder2
        imgl = self.transform(Image.open(self.folder_left[vl])).to(device)
        imgr = self.transform(Image.open(self.folder_right[vl])).to(device)

        # Load and transform uint16 image from folder3
        imgd = Image.open(self.folder_disp[vl])
        imgd = transforms.ToTensor()(imgd).to(device)
        imgd = imgd // 256

        imgd = imgd.squeeze()

        return imgl, imgr, imgd

#device = torch.device("cpu")

model = StereoNet(120)
#model.load_state_dict(torch.load("nets/4000_0.1989159608883035.pth"))
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

base = "E:\DrivingStereo/"
driving_stereo = DrivingStereo(base + "train-left", base + "train-right", base + "train-disp")
train_size = int(0.998 * len(driving_stereo))
test_size = len(driving_stereo) - train_size
train_dataset, test_dataset = random_split(driving_stereo, [train_size, test_size], torch.Generator().manual_seed(0))
dataloaderTrain = DataLoader(train_dataset, batch_size=1, shuffle=True)
dataloaderTest = DataLoader(test_dataset, batch_size=1, shuffle=True)

def invoke_model(left, right, disp):
    max_disp = 120
    until = left.shape[3] - max_disp
    ground_disp = disp[:, :, :(until)]
    ground_disp[ground_disp >= max_disp] = 0
    ground_disp = ground_disp.long()

    pred = model(left, right, max_disp)
    pred_red = pred[:, :, :, :(until)]

    return pred_red, ground_disp

model.train()
count = 0
for i in range(2):
    for left, right, disp in dataloaderTrain:

        pred_red, ground_disp = invoke_model(left, right, disp)
        loss = F.cross_entropy(pred_red, ground_disp, ignore_index=0, label_smoothing=0.1)

        print(f"Train loss: {loss.item()}, count: {count}", end="\r")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if count % 1000 == 0 and count > 0:
            model.eval()
            with torch.no_grad():
                acc_sum = 0
                for left, right, disp in tqdm.tqdm(dataloaderTest, "Test"):
                    pred_red, ground_disp = invoke_model(left, right, disp)
                    predicted_labels = torch.argmax(pred_red, dim=1)
                    defined_mask = ground_disp != 0
                    correct_predictions = (predicted_labels[defined_mask] - ground_disp[defined_mask] < 3).sum().item()
                    num_defined = (defined_mask).numel()
                    accuracy = correct_predictions / num_defined
                    acc_sum += accuracy
                    torch.cuda.empty_cache()

                accuracy = acc_sum / len(test_dataset)
                print(f"Accuracy: {accuracy}")
            torch.save(model.state_dict(), f"nets/{count}_{accuracy}.pth")
            model.train()

        count += 1
        torch.cuda.empty_cache()
