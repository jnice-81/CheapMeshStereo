import stereonet
import torch
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF

if False: # Test
    model = stereonet.StereoNet()
    model.load_state_dict(torch.load("nets/stereo_net/standard_matching_best(71%).pth"))
    device = torch.device('cuda')

    transform = transforms.Compose([
                transforms.ToTensor(), transforms.Lambda(lambda x: x[0:3, :, :]), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    imgl = transform(Image.open("imgl.jpg")).unsqueeze(0).to(device)
    imgr = transform(Image.open("imgr.jpg")).unsqueeze(0).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        disp, var = model(imgl, imgr, 120)

    TF.to_pil_image(disp / 120).show()
    TF.to_pil_image(var / 50).show()

if True:
    from torch.utils.mobile_optimizer import optimize_for_mobile

    model = stereonet.StereoNet()
    model.load_state_dict(torch.load("nets/stereo_net/standard_matching_best(71%).pth"))
    model.eval()

    exampleL = torch.rand(1, 3, 400, 400)
    exampleR = torch.rand(1, 3, 400, 400)

    traced = torch.jit.trace(model, [exampleL, exampleR])
    traced_optim = optimize_for_mobile(traced)
    traced_optim.save("stereo_net.ptl")

