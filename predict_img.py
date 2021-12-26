import argparse
import cv2
import torch
from network import ResnetUnetHybrid
import image_utils
from pathlib import Path


def predict_img(args):
    """Inference a single image."""
    # switch to CUDA device if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # load model
    print('Loading model...')
    model = ResnetUnetHybrid.load_pretrained(device=device, load_path=args.model_path)
    model.eval()

    images_path = Path(args.imgs_path)
    output_folder = Path(args.imgs_path + "/output/")
    output_folder.mkdir(parents=True, exist_ok=True)

    for img_path in images_path.glob('*'):
        if img_path.is_dir():
            continue
        # load image
        img = cv2.imread(img_path)[..., ::-1]
        img = image_utils.scale_image(img)
        img = image_utils.center_crop(img)
        inp = image_utils.img_transform(img)
        inp = inp[None, :, :, :].to(device)

        # inference
        print('Running the image through the network...')
        output = model(inp)

        # transform and plot the results
        output = output.cpu()[0].data.numpy()
        image_utils.show_img_and_pred(img, output)
        image_utils.save_pred(output, output_folder/img_path.name)
    


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_path', required=True, type=str, help='Path to the input image.')
    parser.add_argument('--model_path', required=True, type=str, help='Path to the model')
    return parser.parse_args()


def main():
    args = get_arguments()
    predict_img(args)


if __name__ == '__main__':
    main()