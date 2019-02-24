import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train import load_model
import json
import argparse

# Define command line arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--image', type=str, help='Image to predict')
arg_parser.add_argument('--checkpoint', type=str, help='Model checkpoint to use when predicting')
arg_parser.add_argument('--topk', type=int, help='Return top K predictions')
arg_parser.add_argument('--labels', type=str, help='JSON file containing label names')
arg_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = arg_parser.parse_known_args()

# Processes Image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns an Numpy array
    '''
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])    
    
    width, height = image.size
    size = 224
    
    if  height > width:
        height = int(max(height * size / width, 1))
        width = int(size)
    else:
        width = int(max(width * size / height, 1))
        height = int(size)
        
    resized_image = image.resize((width, height))
        
    x0 = (width - size) / 2
    y0 = (height - size) / 2
    x1 = x0 + size
    y1 = y0 + size
    
    image_c = image.crop((x0, y0, x1, y1))
    image_array = np.array(image_c) / 255
    
    image_array_n = (image_array - mean) / std
    image_array_n = image_array_n.transpose((2, 0, 1))
    
    return image_array_n

# Implement the code to predict the class from an image file
def predict(image, checkpoint, topk=5, labels='', gpu=True):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Use command line values when specified
    if args.image:
        image = args.image     
        
    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk
            
    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu
    
    # Loading the checkpoint
    checkpoint_dict = torch.load(checkpoint)
    
    arch = checkpoint_dict['arch']
    output_size = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
    
    model = load_model(arch, output_size, hidden_units)
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
    
    if gpu: 
        model.to('cuda')
    
    model.eval()
    image = Image.open(image_path)
    image_array = process_image(image)
    image_tensor = torch.from_numpy(image_array)
    
    inputs = image_tensor.type(torch.cuda.FloatTensor)
    inputs = inputs.unsqueeze(0)
    
    # Forward Pass on Model
    output = model.forward(inputs)  
    ps = torch.exp(output).data.topk(topk)
    
    ps_top_five = ps[0].cpu()
    classes = ps[1].cpu()
    
    class_to_idx_i = {model.class_to_idx[i]: i for i in model.class_to_idx}
    
    classes_m = [class_to_idx_i[label] for label in classes.numpy()[0]]
    
    return ps_top_five.numpy()[0], classes_m

# Make predictions if invoked from command line
if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)
    
    
    