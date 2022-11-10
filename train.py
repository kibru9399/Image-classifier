from for_model import *
from load_utility import *
import argparse

parser = argparse.ArgumentParser(description = 'create and train a model')

parser.add_argument("data_dir", type=str, help = 'directory of the data')
parser.add_argument("--save_dir", type=str, help = 'directory for trained model')
parser.add_argument("--arch", type=str, help = 'Architucture of the model')
parser.add_argument("--learning_rate", type=float, help = 'learning rate of the model')
parser.add_argument("--device", type=str, help = 'cpu or gpu')
parser.add_argument("--epochs", type=int, help = 'Number of epochs to train the model')
args = parser.parse_args()



def create_model(data_dir, save_dir, arch, learning_rate, device, epochs):
    train_loader, valid_loader, test_loader, train_images = transformer(data_dir)
    model = classifier()
    model.create_my_model(arch, learning_rate)
    model.train(train_loader, valid_loader, train_images, save_dir, device, epochs)
if __name__ == '__main__':
    create_model(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.device, args.epochs)