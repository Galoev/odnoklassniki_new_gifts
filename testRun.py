import matplotlib
import argparse
from PIL import Image
from pathlib import Path
import torch
import matplotlib.pyplot as plt


import numpy as np

parser = argparse.ArgumentParser(description='PyTorch depth prediction evaluation script')
parser.add_argument('--model_folder', type=str, metavar='F',
                    help='In which folder have you saved the models')
parser.add_argument('--model_name', type=str, metavar='F',
                    help='Name of the model')
parser.add_argument('--model_no', type=int, default = 1, metavar='N',
                    help='Which model no to evaluate')
parser.add_argument('--path', type=str, default='data', metavar='D',
                    help="image file path")
parser.add_argument('--model_type', type=str, default='unet',
                    help='In which folder have you saved the models')
args = parser.parse_args()

from data import output_height, output_width

state_dict = torch.load(args.model_folder + args.model_name + "/model_" + str(args.model_no) + ".pth")

if args.model_type == "unet":
  from unet import UNet
  print("Import unet")
elif args.model_type == "tiny_unet":
  from tiny_unet import UNet
  print("Import tiny_unet")
else:
  print("Error. Unknown type model")
  exit(0)
model = UNet()

model.load_state_dict(state_dict)
model.eval()

res_folder = Path(args.path + "/output_" + args.model_name + "_" + args.model_no)
res_folder.mkdir(parents=True, exist_ok=True)

images_path = Path(args.path)
for file in images_path.glob('*.h5'):
    img = Image.open(str(file.resolve()))
    img = img.resize((64,64))
    img_np = np.asarray(img)
    img_t = torch.from_numpy(img_np)
    img_t = img_t.view(1, 3, output_height, output_width)
    img_t = img_t.float()
    output = model(img_t)
    output = output.detach().numpy()
    print(output.shape)
    plt.imsave(str(res_folder) + "/" +f"output_{args.model_name}_{args.model_no}.png", np.transpose(output[0][0], (0, 1)))
