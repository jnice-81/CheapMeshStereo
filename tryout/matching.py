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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ResidualBlock, self).__init__()
        pad = (3 + 2 * (dilation - 1) -1) // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out

class FeatureNetwork(nn.Module):
    def __init__(self, num_downsampling_layers = 3, num_residual_blocks = 6, in_channels=3, out_channels=32):
        super(FeatureNetwork, self).__init__()
        layers = []
        # Downsample the input images using K 5x5 convolutions with a stride of 2
        for _ in range(num_downsampling_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        # Apply residual blocks
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dilation=1))

        # Final layer with a 3x3 convolution without batch normalization or activation
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        self.feature_network = nn.Sequential(*layers)

    def forward(self, x):
        return self.feature_network(x)
    
class CostVolume(nn.Module):
    def __init__(self, num_filtering_layers=4):
        super(CostVolume, self).__init__()

        conv3d_layers = []
        filter_channels = 32
        for _ in range(num_filtering_layers):
            conv3d_layers.append(nn.Conv3d(filter_channels, filter_channels, kernel_size=3, stride=1, padding=1))
            conv3d_layers.append(nn.BatchNorm3d(filter_channels))
            conv3d_layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        conv3d_layers.append(nn.Conv3d(filter_channels, 1, kernel_size=3, stride=1, padding=1))
        self.filtering = nn.Sequential(*conv3d_layers)
        self.feature_extraction = FeatureNetwork()

    def forward(self, l, r, num_candidates):
        fl = self.feature_extraction(l)
        fr = self.feature_extraction(r)

        cost = torch.ones((fl.shape[0], 32, num_candidates, fl.shape[2], fl.shape[3]), device=fl.device)
        for d in range(num_candidates):
            until = fl.shape[3] - d
            cost[:, :, d, :, :until] = fl[:, :, :, d:] - fr[:, :, :, :until]

        cost_filtered = self.filtering(cost)

        exp_cost = torch.exp(-cost_filtered)
        cost_all = torch.sum(exp_cost, dim=2, keepdim=True)
        prob = exp_cost / cost_all
        index_tensor = torch.arange(cost_filtered.shape[2], dtype=torch.float32, device=cost_filtered.device).reshape((1, 1, -1, 1, 1))
        cost_mul = index_tensor * prob
        disp = torch.sum(cost_mul, dim=2)

        var = prob * (index_tensor - disp) ** 2
        var = F.interpolate(disp, size=l.shape[-2:], mode='bilinear')
        return disp, var

class RefinementNetwork(nn.Module):
    def __init__(self):
        super(RefinementNetwork, self).__init__()

        conv_input = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)

        residual_blocks = []
        dilation_factors = [1, 2, 4, 8, 1, 1]
        for dilation in dilation_factors:
            residual_blocks.append(ResidualBlock(32, 32, dilation))

        conv_output = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.refinenet = nn.Sequential(conv_input, *residual_blocks, conv_output)

    def forward(self, disparity_input, color_input):
        input_concatenated = torch.cat([color_input, disparity_input], dim=1)

        out = self.refinenet(input_concatenated)

        return F.relu(out + disparity_input)
    
class StereoNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cvol = CostVolume(3)
        self.ref_net = nn.ModuleList()
        self.relus = nn.ModuleList()

        for i in range(3):
            self.ref_net.append(RefinementNetwork())
            self.relus.append(nn.ReLU())

    def forward(self, l, r, num_candiates):
        disp, var = self.cvol(l, r, num_candiates // (2 ** len(self.ref_net)))
        orig_shape = l.shape[-2:]

        for i in range(len(self.ref_net) - 1, -1, -1):
            divby = (2 ** i)
            current_shape = (orig_shape[0] // divby, orig_shape[1] // divby)
            scale_up = current_shape[-1] / disp.shape[-1]
            disp = disp * scale_up
            disp = F.interpolate(disp, size=current_shape, mode='bilinear')
            linp = F.interpolate(l, size=current_shape, mode='bilinear')
            
            disp = self.ref_net[i](disp, linp)

        if self.training:
            return disp.squeeze(1)
        else:
            return disp.squeeze(1), var.squeeze(1)

    
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

        imgl = self.transform(Image.open(self.folder_left[vl])).to(device)
        imgr = self.transform(Image.open(self.folder_right[vl])).to(device)

        # Load and transform uint16 image from folder3
        imgd = Image.open(self.folder_disp[vl])
        imgd = transforms.ToTensor()(imgd).to(device)
        imgd = imgd // 256

        imgd = imgd.squeeze()

        return imgl, imgr, imgd

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

max_disp = 120
model = StereoNet()
#num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#print(f"Number of trainable parameters: {num_params}")
model.load_state_dict(torch.load("nets/4000_0.6053308844566345.pth"))
model.to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.9)

base = "E:\DrivingStereo/"
driving_stereo = DrivingStereo(base + "train-left", base + "train-right", base + "train-disp")
train_size = int(0.998 * len(driving_stereo))
test_size = len(driving_stereo) - train_size
train_dataset, test_dataset = random_split(driving_stereo, [train_size, test_size], torch.Generator().manual_seed(0))
dataloaderTrain = DataLoader(train_dataset, batch_size=1, shuffle=True)
dataloaderTest = DataLoader(test_dataset, batch_size=1, shuffle=True)

def invoke_model(left, right, disp, train=True):
    until = left.shape[3] - max_disp
    ground_disp = disp[:, :, :(until)]
    ground_disp[ground_disp >= max_disp] = 0
    ground_disp = ground_disp.type(torch.float32)

    if model.training:
        preds = model(left, right, max_disp)
    else:
        preds, var = model(left, right, max_disp)
        var = var[:, :, :(until)]
    pred_red = preds[:, :, :(until)]

    if model.training:
        return pred_red, ground_disp
    else:
        return pred_red, ground_disp, var

model.train()
count = 0
for i in range(2):
    for left, right, disp in dataloaderTrain:
        pred_red, ground_disp = invoke_model(left, right, disp)
        defined_mask = ground_disp != 0
        loss = torch.sum((((((pred_red[-1] - ground_disp)[defined_mask] / 2) ** 2 + 1) ** 0.5) - 1) / (torch.sqrt(torch.tensor(2.0)) - 1))

        print(f"Train loss: {loss.item()}, count: {count}", end="\r")

        #loss.backward()
        #optimizer.step()
        #optimizer.zero_grad()

        if True or count % 1000 == 0 and count > 0:
            model.eval()
            with torch.no_grad():
                acc_sum = 0
                for left, right, disp in tqdm.tqdm(dataloaderTest, "Test"):
                    pred_red, ground_disp, var = invoke_model(left, right, disp, False)
                    mean_var = var.median()
                    #pred_red[var > mean_var] = 0
                    TF.to_pil_image(ground_disp / 120).show()
                    TF.to_pil_image(pred_red / 120).show()
                    #TF.to_pil_image(var).show()
                    defined_mask = ground_disp != 0
                    correct_predictions = (torch.abs(pred_red[defined_mask] - ground_disp[defined_mask]) <= 1).sum().item()
                    num_defined = torch.sum(defined_mask)
                    accuracy = correct_predictions / num_defined
                    acc_sum += accuracy

                accuracy = acc_sum / len(test_dataset)
                print(f"Accuracy: {accuracy}")
            torch.save(model.state_dict(), f"nets/{count}_{accuracy}.pth")

            model.train()
            scheduler.step()

        count += 1
        #torch.cuda.empty_cache()
