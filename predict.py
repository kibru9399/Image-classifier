from formodel import *
from load_utility import *
import argparse
import re

parser = argparse.ArgumentParser(description = 'predict by loading the model')

parser.add_argument("image_path", type=str, help = 'directory of the image to be predicted')
parser.add_argument("check_point", type=str, help = 'directory for saved checkpoint')
parser.add_argument("--top_k", type=int, help = 'how many classes to show')
args = parser.parse_args()

def predict(image_path, check_point, top_k):
    model = load_checkpoint(check_point)  #loading the checkpoint 
    
    
    prop, cls = classifier().predict(image_path, model, top_k)
    cls = [cat_to_name[cls] for cls in cls]
    
    label =re.search('(?<=\/)([0-9]*)\/', image_path).group(1)     
    flower_name = cat_to_name[str(label)]
    print('The real name of the flower is : ', flower_name)
    print('The predicted classes and their probabilitis are : ', cls, prop)
 
if __name__ == '__main__':
    predict(args.image_path, args.check_point, args.top_k)