import cv2
import os
import json
import numpy as np

base = "3dmodels/old/corner"
folder_path = f'{base}/images/'
image_list = []
test_image_list = []
test_cam_list = []
cam_list = []
pointcloud = []
image_to_points = {}
points_to_image = {}

seen_points = {}
test_indices = set()

with open(f"{base}/ARCoreData.json") as f:
    direct = json.load(f)["ARCoreData"]
coredata = {}
for i, core in enumerate(direct):
    coredata[core["name"]] = core

    image_path = os.path.join(folder_path, f"{core['name']}.jpg")
    
    image = cv2.imread(image_path)

    project = np.array(core["projection"]).reshape((4, 4), order="F")
    view = np.array(core["viewmatrix"]).reshape((4, 4), order="F")
    
    P = project @ view
    Pinv = np.linalg.inv(P)

    Pinv = Pinv.astype(np.float32)
    P = P.astype(np.float32)
    view = view.astype(np.float32)
    project = project.astype(np.float32)
    

    if i in test_indices:
        test_image_list.append(image)
        test_cam_list.append((Pinv, P, view, project))
    else:
        image_list.append(image)
        cam_list.append((Pinv, P, view, project))

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

idx1 = 8
idx2 = 9

"""
common = set()
common.update(image_to_points[idx1])
common.intersection_update(image_to_points[idx2])

points3d = [pointcloud[i] for i in common]

def pback(P, point):
    point = np.append(point, 1)
    t = P @ point
    t /= t[3]
    t = np.array([t[0], t[1]])
    t[0] = (t[0] + 1) * 0.5 * image_list[0].shape[1]
    t[1] = (1 - t[1]) * 0.5 * image_list[0].shape[0]
    return t

points2d1 = [pback(cam_list[idx1][1], p) for p in points3d]
points2d2 = [pback(cam_list[idx2][1], p) for p in points3d]

points3d = np.array([p[0:3] for p in points3d], np.float32)
points2d1 = np.array(points2d1, np.float32)
points2d2 = np.array(points2d2, np.float32)

    # Define camera calibration flags
calibration_flags = (
    cv2.CALIB_FIX_INTRINSIC
)

A = np.array([
    [905.353455, 0, 358.192932],
    [0, 910.3363, 639.527954],
    [0, 0, 1]
], np.float32)

# Stereo calibration
ret, camera_matrix1, dist_coeffs1, camera_matrix2, dist_coeffs2, R, T, E, F = cv2.stereoCalibrate(
    [points3d], [points2d1], [points2d2],
    A, None, A, None,
    imageSize=(image_list[0].shape[1], image_list[0].shape[0]),
    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000000000, 1e-10),
    flags=calibration_flags
)

# Print the results
print("RET")
print(ret)
print("Camera Matrix 1:")
print(camera_matrix1)
print("Distortion Coefficients 1:")
print(dist_coeffs1)
print("Camera Matrix 2:")
print(camera_matrix2)
print("Distortion Coefficients 2:")
print(dist_coeffs2)
print("Rotation Matrix:")
print(R)
print("Translation Vector:")
print(T)
"""

# See http://www.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche0092.html
G = np.array([
    [1, 1, 1, 1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [0, 0, 0, 1]
])
V1 = G * cam_list[idx1][2]
V2 = G * cam_list[idx2][2]
R1glob = V1[0:3, 0:3]
R2glob = V2[0:3, 0:3]
t1glob = V1[0:3, 3]
t2glob = V2[0:3, 3]
R = R2glob @ R1glob.T
T = t2glob - R @ t1glob

image_width = image_list[0].shape[1]
image_height = image_list[0].shape[0]

Pm1 = cam_list[idx1][3]
camera_matrix1 = np.array([
    [Pm1[0, 0] * 0.5 * image_width, 0, Pm1[0, 2] + image_width * 0.5],
    [0, Pm1[1, 1] * 0.5 * image_height, Pm1[1, 2] + image_height * 0.5],
    [0, 0, 1]
])
camera_matrix2 = camera_matrix1


R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    camera_matrix1, None, camera_matrix2, None,
    (image_width, image_height), R, T, alpha=-1, flags=cv2.CALIB_ZERO_DISPARITY
)

# Compute the rectification mapping and rectify the images
map1_left, map2_left = cv2.initUndistortRectifyMap(
    camera_matrix1, None, R1, P1, (image_width, image_height), cv2.CV_16SC2
)
map1_right, map2_right = cv2.initUndistortRectifyMap(
    camera_matrix2, None, R2, P2, (image_width, image_height), cv2.CV_16SC2
)

# Rectify the images
rectified_left = cv2.remap(image_list[idx1], map1_left, map2_left, cv2.INTER_LINEAR)
rectified_right = cv2.remap(image_list[idx2], map1_right, map2_right, cv2.INTER_LINEAR)

ndisp = 16 * 15
stereo: cv2.StereoBM = cv2.StereoBM_create(numDisparities=ndisp, blockSize=11)
#stereo.setMinDisparity(-16 * 15)
disparity = stereo.compute(cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY))
#disparity: np.array = disparity.astype(np.float32) / 16
disparity[disparity == -16] = 0
disparity = disparity.astype(np.float32) / 16.0

depth = cv2.reprojectImageTo3D(disparity, Q)

im = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.float32)

for rows in range(depth.shape[0]):
    for cols in range(depth.shape[1]):
        td = R1.T @ depth[rows, cols]
        if (np.any(np.isnan(td))):
            continue
        td = np.append(td, 1)
        p1 = V2 @ np.linalg.inv(V1) @ td
        p1 /= p1[3]
        proj = camera_matrix1 @ p1[0:3]
        proj /= proj[2]
        if proj[0] >= 0 and proj[0] < im.shape[1] and proj[1] >= 0 and proj[1] < im.shape[0]:
            im[int(proj[1]), int(proj[0])] = 1.0
        else:
            #print(proj)
            pass


#depth = depth[:, :, 2]

"""
double d = sptr[x];
            Vec4d homg_pt = _Q*Vec4d(x, y, d, 1.0);
            dptr[x] = Vec3d(homg_pt.val);
            dptr[x] /= homg_pt[3];

            if( fabs(d-minDisparity) <= FLT_EPSILON )
                dptr[x][2] = bigZ;
"""

cv2.namedWindow('Disp', cv2.WINDOW_NORMAL)
cv2.namedWindow('q', cv2.WINDOW_NORMAL)
cv2.imshow('Disp', im)
cv2.imshow('q', image_list[idx2])

# Display or save the rectified images
#cv2.imshow('Rectified Left Image', rectified_left)
#cv2.imshow('Rectified Right Image', rectified_right)
while cv2.waitKeyEx(100) != ord('a'):
    pass
cv2.destroyAllWindows()