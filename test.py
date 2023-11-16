import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
from function import calc_mean_std, normal, coral
import models.transformer as transformer
import models.StyTR as StyTR
import matplotlib.pyplot as plt
from matplotlib import cm
from function import normal
import numpy as np
import time
def test_transform(size, crop):
    transform_list = []
   
    if size != 0: 
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
def style_transform(h,w):
    k = (h,w)
    size = int(np.max(k))
    transform_list = []    
    transform_list.append(transforms.CenterCrop((h,w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

def content_transform():
    transform_list = []   
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform

#normalize weight parameter
def normalize(weights):
    denom = sum(weights)
    if denom >0.:
        return [float(i) / denom for i in weights]
    else:
        return [0.] * len(weights)
  
def file_list_to_str(file_paths):
    file_names = []
    for file_path in file_paths:
        base_name = os.path.basename(file_path)
        file_name, _ = os.path.splitext(base_name)
        file_names.append(file_name)

    flattened_string = '_'.join(file_names)
    return flattened_string


parser = argparse.ArgumentParser()
parser.add_argument('--content', type=str,
                    help='File path to the content image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--decoder_path', type=str, default='experiments/decoder_iter_160000.pth')
parser.add_argument('--Trans_path', type=str, default='experiments/transformer_iter_160000.pth')
parser.add_argument('--embedding_path', type=str, default='experiments/embedding_iter_160000.pth')
parser.add_argument('--a', type=float, default=1.0)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
"""
Adding in arguments for multiple styles and weights
"""
parser.add_argument('--style', type=str,
                    help='File path to the style image, or multiple style \
                    images separated by commas if you want to do style \
                    interpolation or spatial control')
parser.add_argument("--style-num", type=int, default=1,
                                  help="number of styles used in training, default is 1")
parser.add_argument('--style_img_weights', type = float, nargs = '+', help = "List of float values of the weighting for each style image.")

args = parser.parse_args()

# Advanced options
content_size=512
style_size=512
crop='store_true'
save_ext='.jpg'
output_path=args.output
preserve_color='store_true'
alpha=args.a

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

if args.content:
    content_paths = [Path(args.content)]

# Either --style or --style_dir should be given.
#modified by me
if args.style:
    style_paths = [Path(p) for p in args.style.split(',')]    

if args.style_img_weights is None or len(args.style_img_weights) != len(style_paths):
    args.style_img_weights = [1.0 / len(args.style) for _ in range(len(args.style))]


#added by me end

if not os.path.exists(output_path):
    os.mkdir(output_path)


vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
Trans = transformer.Transformer()
embedding = StyTR.PatchEmbed()

decoder.eval()
Trans.eval()
vgg.eval()
from collections import OrderedDict
new_state_dict = OrderedDict()
state_dict = torch.load(args.decoder_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
decoder.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.Trans_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
Trans.load_state_dict(new_state_dict)

new_state_dict = OrderedDict()
state_dict = torch.load(args.embedding_path)
for k, v in state_dict.items():
    namekey = k
    new_state_dict[namekey] = v
embedding.load_state_dict(new_state_dict)


"""
this is the code where the images actually get stylized.
"""
network = StyTR.StyTrans(vgg,decoder,embedding,Trans,args) #this is the init
network.eval()
network.to(device)

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

"""The idea here is to remove the for loop of style paths and instead have all the styles sum up"""
#need to have style images as nested tensors
style_tensors = [] 

#Sets up all the style images
for style_img in style_paths:
    h,w,c = np.shape(content_tf(Image.open(content_paths[0]).convert("RGB")))
    style_tf1 = style_transform(h,w) #this is for image sizing: i don't actually think it gets called
    style = style_tf(Image.open(style_img).convert("RGB"))
    style_tensors.append(style.to(device).unsqueeze(0))    

#I made styles into an array so the style loss can be calculated that way.
for content_path in content_paths:
    print(content_path)
    print(f'Styles: {style_paths}')
    content_tf1 = content_transform()       
    content = content_tf(Image.open(content_path).convert("RGB"))

    h,w,c=np.shape(content)    
    content = content.to(device).unsqueeze(0)
    
    with torch.no_grad():
        output= network(content,style_tensors)  
    output = output[0][0].cpu()
    output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
        output_path, splitext(basename(content_path))[0],
        file_list_to_str(style_paths), save_ext
    )
    print(f'Stylizing Complete: Output name: {output_name}')

    save_image(output, output_name)
