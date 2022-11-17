import torch
from torchvision import datasets, transforms
from PIL import Image


    
def transformer(data_dir='flowers'):
    ''' transforms the images and loads them'''
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    #Load the datasets with ImageFolder
    train_images = datasets.ImageFolder(train_dir, transform = train_transforms)
    validation_images = datasets.ImageFolder(valid_dir, transform = validation_transforms)
    test_images = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_images, batch_size = 64, shuffle = True )
    valid_loader = torch.utils.data.DataLoader(validation_images, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size = 64)
    
    return train_loader, valid_loader, test_loader, train_images


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #using same tranformation as the ones used in validation and test sets
    process = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    
    return process(Image.open(image))


def load_checkpoint(filepath):
    '''loads the check point'''
    checkpoint = torch.load(filepath)
    model= checkpoint['model']        #since the checkpoint was saved as a model, not as state_dict, no need to \
                                       #reconstruct a model to load state dict
    
    return model
    