import torch
import torch.nn as nn
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import random
import tqdm

base = "3dmodels/cup"
folder_path = f'{base}/images/'
image_list = []
test_image_list = []
test_cam_list = []
cam_list = []
pointcloud = []
image_to_points = {}
points_to_image = {}

seen_points = {}
test_indices = set([5])

with open(f"{base}/ARCoreData.json") as f:
    direct = json.load(f)["ARCoreData"]
coredata = {}
for i, core in enumerate(direct):
    coredata[core["name"]] = core

    image_path = os.path.join(folder_path, f"{core['name']}.jpg")
    
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, None, fx=0.05, fy=0.05)
    resized_image = torch.tensor(resized_image, dtype=torch.float32) /  255

    project = np.array(core["projection"]).reshape((4, 4), order="F")
    view = np.array(core["viewmatrix"]).reshape((4, 4), order="F")
    
    P = project @ view
    Pinv = np.linalg.inv(P)
    pos = np.linalg.inv(view)[:, 3]
    pos /= pos[3]

    Pinv = Pinv.astype(np.float32)
    P = P.astype(np.float32)
    pos = pos.astype(np.float32)

    if i in test_indices:
        test_image_list.append(resized_image)
        test_cam_list.append((Pinv, P, pos))
    else:
        image_list.append(resized_image)
        cam_list.append((Pinv, P, pos))

        image_idx = len(image_list) - 1
        image_to_points[image_idx] = []
        for id, pos in zip(core["pointIDs"], core["pointPos"]):
            if not (id in seen_points):
                pointcloud.append(pos)
                point_idx = len(pointcloud) - 1
                seen_points[id] = point_idx
                points_to_image[point_idx] = [] # Point has not been seen so far
                points_to_image[point_idx].append(image_idx)
                image_to_points[image_idx].append(point_idx)
            else:
                point_idx = seen_points[id]
                points_to_image[point_idx].append(image_idx)
                image_to_points[image_idx].append(point_idx)
                

class Nerf(nn.Module):
    def __init__(self, load_weights) -> None:
        super().__init__()
        
        self.hashmaps = [(50, 32, 1), (200, 16, 0.5), (400, 8, 0.25), (400, 4, 0.125), (400, 3, 0.05)]

        self.embeddings = nn.ModuleList()
        concat_size = 0
        for hsize, shape, _ in self.hashmaps:
            self.embeddings.append(nn.Embedding(hsize, shape))
            concat_size += shape
        
        self.densityLayer = nn.Sequential()
        self.densityLayer.append(nn.Linear(concat_size, concat_size))
        self.densityLayer.append(nn.ReLU())
        self.densityLayer.append(nn.Linear(concat_size, 1))
        self.colorLayer = nn.Sequential()
        self.colorLayer.append(nn.Linear(concat_size, concat_size))
        self.colorLayer.append(nn.ReLU())
        self.colorLayer.append(nn.Linear(concat_size, 3))
        
        if load_weights:
            self.load_state_dict(torch.load("weights"))

    def get_encoded_pos(self, pos, voxel_sidelength, level):
        pos = pos / voxel_sidelength

        selects = torch.zeros((pos.shape[0] * 8, 3), dtype=torch.int32)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    idx = i * 4 + j * 2 + k
                    for mp, g in enumerate([i, j, k]):
                        if g == 0:
                            selects[idx::8, mp] = torch.floor(pos[:, mp])
                        else:
                            selects[idx::8, mp] = torch.ceil(pos[:, mp])

        hashed = torch.bitwise_xor(selects[:, 0] * 73856093, selects[:, 1] * 19349663)
        hashed = torch.bitwise_xor(hashed, selects[:, 2] * 83492791)
        hashed = hashed % self.hashmaps[level][0]
        hashed = hashed.reshape((-1, 1))

        emb = self.embeddings[level](hashed)
        emb = emb.reshape((-1, emb.shape[-1]))
        pos_rem = pos - torch.floor(pos)
        
        m1 = torch.tensor([True, True, True, True, False, False, False, False]).repeat(pos.shape[0])
        pr1 = torch.repeat_interleave(pos_rem[:, 0], 4).unsqueeze(1)
        lin1 = pr1 * emb[m1, :] + (1 - pr1) * emb[torch.logical_not(m1), :]

        m2 = torch.tensor([True, True, False, False]).repeat(pos.shape[0])
        pr2 = torch.repeat_interleave(pos_rem[:, 1], 2).unsqueeze(1)
        lin2 = pr2 * lin1[m2, :] + (1 - pr2) * lin1[torch.logical_not(m2), :]

        m3 = torch.tensor([True, False]).repeat(pos.shape[0])
        pr3 = pos_rem[:, 2].unsqueeze(1)
        lin3 = pr3 * lin2[m3, :] + (1 - pr3) * lin2[torch.logical_not(m3), :]

        return lin3


    def forward(self, x):
        values = []
        for i, (_, _, voxel_sidelength) in enumerate(self.hashmaps):
            values.append(self.get_encoded_pos(x, voxel_sidelength, i))

        f = torch.concat(values, dim=1)
        
        dense = self.densityLayer(f)
        #fd = torch.concat((f, d), dim=1)
        color = self.colorLayer(f)

        return (dense, color)
    
    def render_patch(self, left, right, top, bottom, width, height,
                    cam, num_steps, step_size):
        rays = self.get_rays(left, right, top, bottom, width, height, cam)
        campos = cam[2]
        rays = torch.tensor(rays)

        ny = rays.shape[0]
        nx = rays.shape[1]
        rays_flat = torch.reshape(rays, (ny*nx, 4))
        patch = torch.zeros((ny * nx, 3))
        T = torch.zeros((ny * nx, 1))
        for i in range(0, num_steps):
            mul = 1 + i * step_size + torch.rand(1) * step_size
            pos = rays_flat * mul + np.reshape(campos, (1, 4))
            pos = pos[:, :3]

            dense, color = self.forward(pos)
            current_dense = dense * (1 / num_steps)
            factor = torch.exp(-T) * (1 - torch.exp(-current_dense))
            patch += factor * color
            T += current_dense

        patch = torch.reshape(patch, (ny, nx, 3))
        return patch
    
    def train(self, cam_list, img_list, test_cam_list, test_img_list, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003)

        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.98, verbose=True)

        #self.load_state_dict(torch.load("weights"))

        for epoch in range(epochs):
            losses = []
            
            optimizer.zero_grad()

            loss = torch.zeros(1)
            for img, cam in zip(img_list, cam_list):
                """
                for _ in range(5):
                    j = random.randint(0, len(cam_list) - 1)
                    img = img_list[j]
                    cam = cam_list[j]

                    x_patch = np.int32(np.random.uniform(0, img.shape[1], 1)).repeat(2)
                    y_patch = np.int32(np.random.uniform(0, img.shape[0], 1)).repeat(2)
                    x_patch = np.clip(x_patch + np.array([-20, 20]), 0, img.shape[1] - 1)
                    y_patch = np.clip(y_patch + np.array([-20, 20]), 0, img.shape[0] - 1)
                """
                
                patch = self.render_patch(0, img.shape[1], 0, img.shape[0], img.shape[1], img.shape[0],
                                        cam, 250, 0.04)
                loss = loss + torch.sum(torch.square(patch - img))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            print(f"Final: {torch.tensor(losses).mean()}, epoch {epoch}")
            if optimizer.param_groups[0]['lr'] > 0.0001:
                scheduler.step()
            
            with torch.no_grad():
                for cam, img in zip(test_cam_list, test_image_list):

                    test = self.render_patch(0, img.shape[1], 0, img.shape[0], img.shape[1], img.shape[0],
                                             cam, 250, 0.04)
                    loss = torch.sum(torch.square(test - img))

                    print(f"Test {loss}")


            torch.save(self.state_dict(), "weights")

        """
        with torch.no_grad():
            cam = cam_list[0]
            img = img_list[0]
            test = self.render_patch(0, img.shape[1], 0, img.shape[0], img.shape[1], img.shape[0], cam)
            cv2.imshow("gen", test.detach().numpy())
            cv2.imshow("ground", (torch.tensor(img, dtype=torch.float32) / 255).numpy())
            cv2.waitKey(0)
        """



    def get_rays(self, left, right, top, bottom, width, height, cam):
        Pinv = cam[0]
        pos = cam[2]
        
        convX = np.array([left, right], dtype=np.float32)
        convX = 2 * (convX / width) - 1
        convY = np.array([top, bottom], dtype=np.float32)
        convY = 1 - 2 * (convY / height)

        ny = bottom - top
        nx = right - left

        y_values = np.linspace(convY[0], convY[1], ny, dtype=np.float32)
        x_values = np.linspace(convX[0], convX[1], nx, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        zeros = np.zeros((ny, nx), dtype=np.float32)
        ones = np.ones((ny, nx), dtype=np.float32)
        inputs_2d = np.stack((x_grid, y_grid, zeros, ones), axis=-1)

        """
        Magic code to get a ray
        rayP = Pinv @ np.array([0.5, 0.5, 0, 1])
        rayP /= rayP[3]
        pos = np.linalg.inv(view)[:, 3]
        pos /= pos[3]
        ray = rayP - pos

        pn = pos + ray * 5
        pn[3] = 1
        v = P @ pn
        v /= v[3]
        """

        flat = np.reshape(inputs_2d, (ny * nx, 4))
        flat = (Pinv @ flat.T).T
        flat /= np.reshape(flat[:, 3], (flat.shape[0], 1))
        flat = flat - np.reshape(pos, (1, 4))

        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        flat = flat / norms
        result_array = np.reshape(flat, (ny, nx, 4))

        return result_array
        

nerf = Nerf(True)
nerf.train(cam_list, image_list, test_cam_list, test_image_list, 300)

"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, c in enumerate(cam_list):
    rays = nerf.get_rays(0, 3, 0, 5, 3, 5, c)
    pos = c[2]
    pos = np.tile(pos, (3*5, 1))

    rays = np.reshape(rays, (3 * 5, 4))
    if i == 0:
        rays *= 15


    if i == 0:
        for j in range(rays.shape[0]):
            ax.quiver(pos[j, 0], pos[j, 1], pos[j, 2], rays[j, 0], rays[j, 1], rays[j, 2], label=f"{j}", color=np.random.rand(3))
        ax.legend()
    else:
        quiver = ax.quiver(pos[:, 0], pos[:, 1], pos[:, 2], rays[:, 0], rays[:, 1], rays[:, 2], color='b')

    w = 1.5
    ax.set_xlim([-w, w])
    ax.set_ylim([-w, w])
    ax.set_zlim([-w, w])
    

test = np.array(test)
ax.scatter(test[:, 0], test[:, 1], test[:, 2], c='black', marker='o')

plt.show()
"""