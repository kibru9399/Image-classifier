from for_model import *
from load_utility import *
import argparse

parser = argparse.ArgumentParser(description = 'create and train a model')

parser.add_argument("data_dir", type=str, help = 'directory of the data')
parser.add_argument("--save_dir", type=str, default = '',  help = 'directory to save the trained model')
parser.add_argument("--arch", type=str, default = 'vgg16', help = 'choose architecture from vgg, resnet, alexnet, or densenet')
parser.add_argument("--learning_rate", type=float, default = 0.001, help = 'learning rate of the model')
parser.add_argument("--hidden_state", type=int, default = 205, help = 'number of nuerons')
parser.add_argument("--device", type=str, default = 'cuda', help = 'cpu or cuda')
parser.add_argument("--epochs", type=int, default = 1, help = 'Number of epochs to train the model')
args = parser.parse_args()



def create_model(data_dir, save_dir, arch,  hidden_state, device, epochs, learning_rate):
    train_loader, valid_loader, test_loader, train_images = transformer(data_dir)
    model = classifier()
    model.create_my_model(arch, learning_rate, hidden_state)
    model.train(train_loader, valid_loader, train_images, save_dir, device, epochs)
if __name__ == '__main__':
    create_model(args.data_dir, args.save_dir, args.arch, args.hidden_state, args.device, args.epochs,  args.learning_rate)