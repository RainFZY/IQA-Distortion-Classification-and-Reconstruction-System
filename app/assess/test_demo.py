"""
Test Demo
    ```bash
    python test_demo.py --im_path=data/I03_01_1.bmp
    ```
 Date: 2020/6/7
"""

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from app.assess.IQADataset import NonOverlappingCropPatches
from app.assess.models import resnet, vgg, lenet5, cnn
import os

path = os.path.dirname(os.path.abspath(__file__))

def test(image_path = "../test_images/blur.jpg"):
    parser = ArgumentParser(description='PyTorch CNNIQA test demo')

    # parser.add_argument("--im_path", type=str, default='../test_images/blur.jpg',
    #                     help="image path")
    parser.add_argument("--model_file", type=str, default=path + '/models/resnet18-LIVE',
                        help="model file (default: models/resnet18-LIVE)")

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    args.im_path = image_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_file == path + '/models/CNNIQA-LIVE':
        model = cnn.CNNIQAnet(ker_size=7,
                          n_kers=50,
                          n1_nodes=800,
                          n2_nodes=800).to(device)
    if args.model_file == path + '/models/resnet18-LIVE':
        model = resnet.ResNet18().to(device)
    if args.model_file == path + '/models/resnet34-LIVE':
        model = resnet.ResNet34().to(device)
    if args.model_file == path + '/models/lenet5-LIVE':
        model = lenet5.LeNet5().to(device)
    if args.model_file == path + '/models/vgg19-LIVE':
        model = vgg.VGG('VGG19').to(device)

    model.load_state_dict(torch.load(args.model_file))

    img = Image.open(args.im_path).convert('L')
    patches = NonOverlappingCropPatches(img, 32, 32)

    model.eval()
    with torch.no_grad():
        try:
            patch_scores = model(torch.stack(patches).to(device))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        print(patch_scores.mean())
        print(patch_scores.mean().item())
        score5 = round(patch_scores.mean().item(), 2)

    # CNNIQAnet
    with torch.no_grad():
        model = cnn.CNNIQAnet(ker_size=7,
                              n_kers=50,
                              n1_nodes=800,
                              n2_nodes=800).to(device)
        args.model_file = path + '/models/CNNIQA-LIVE'
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        try:
            patch_scores = model(torch.stack(patches).to(device))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        print(patch_scores.mean().item())
        score1 = round(patch_scores.mean().item(), 2)

    # lenet5
    with torch.no_grad():
        model = lenet5.LeNet5().to(device)
        args.model_file = path + '/models/lenet5-LIVE'
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        try:
            patch_scores = model(torch.stack(patches).to(device))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        print(patch_scores.mean().item())
        score2 = round(patch_scores.mean().item(), 2)

    # vgg19
    with torch.no_grad():
        model = vgg.VGG('VGG19').to(device)
        args.model_file = path + '/models/vgg19-LIVE'
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        try:
            patch_scores = model(torch.stack(patches).to(device))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        print(patch_scores.mean().item())
        score3 = round(patch_scores.mean().item(), 2)

    # resnet34
    with torch.no_grad():
        model = resnet.ResNet34().to(device)
        args.model_file = path + '/models/resnet34-LIVE'
        model.load_state_dict(torch.load(args.model_file))
        model.eval()
        try:
            patch_scores = model(torch.stack(patches).to(device))
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
        print(patch_scores.mean().item())
        score4 = round(patch_scores.mean().item(), 2)

    ave_score = round((score1 + score2 + score3 + score4 + score5) / 5.0, 2)

    # f = open('result.txt', 'w')
    # f.write(str(patch_scores.mean().item()))
    # f.close()
    return [str(score1), str(score2), str(score3), str(score4), str(score5), str(ave_score)]

if __name__ == "__main__":
    test("D:/Python Studio/IQA/考核/www/static/img/blur.jpg")