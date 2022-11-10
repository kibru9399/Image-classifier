# Imports here

import torchvision
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from load_utility import*
    
#creating a classifier class

class classifier:
    def __init__(self):
        """initializing the atributes of a model"""
        self.model = None              #the full architecture of the model
        self.criterion = None
        self.optimizer = None
        self.arch = None              # architecture of the model desired by user
        self.learning_rate = None
        self.epochs = None
        self.device = None            # device to train the model on, similar to address 
        self.save_dir = None          #storge location for the trained model
    def create_my_model(self, arch, learning_rate):
        " This method will create the full trainable architecture along with an optimizer and a loss function"""
        self.arch = arch       
        self.learning_rate = learning_rate
        self.model = getattr(torchvision.models, self.arch)(pretrained=True)     #to choose the architecture such as vgg16
        for param in self.model.parameters():
            param.requires_grad = False
        classifier_input = self.model.classifier[0].in_features    #to get the number of output from the features part of the model
        self.model.classifier = nn.Sequential(nn.Linear(classifier_input, 200),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(200, 150),
                                nn.ReLU(),
                                nn.Linear(150, 120),
                                nn.ReLU(),
                                nn.Linear(120, 102),
                                nn.LogSoftmax(dim=1))
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = self.learning_rate)

    def train(self,train_loader, valid_loader, train_images, save_dir, device, epochs):
        """A method to train the above defined model by taking the datasets, and save checkpoint by taking the directory"""
        self.epochs = epochs
        self.save_dir = save_dir
        self.device = device
        self.model.to(device)
        epochs = epochs
        batch_num = 0

        print_every = 5

        for epoch in range(self.epochs):
            running_loss = 0
            for inputs, labels in train_loader:
                batch_num += 1
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                if batch_num % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()

                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            logps = self.model.forward(inputs)
                            batchloss = self.criterion(logps, labels)
                            test_loss += batchloss.item()

                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim = 1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"validation loss: {test_loss/len(valid_loader):.3f}.. "
                          f"validation accuracy: {accuracy/len(valid_loader):.3f}")
                    running_loss = 0
                    self.model.train()        
        
        self.model.class_to_idx = train_images.class_to_idx
        checkpoint = {
            'class_to_idx' : self.model.class_to_idx,
            'criterion': self.criterion,
            'optimizer': self.optimizer,
            'model': self.model}
        torch.save(checkpoint, self.save_dir + 'checkpoint1.pth')
        
    def predict(self, image_path, model, topk):
        ''' Predict the class (or classes) of an image using a trained model.
        '''
        device = 'cuda'
        image = process_image(image_path)
        image = image.view(1, *image.shape).to(device)

        with torch.no_grad():
            logps = model.forward(image)
        ps = torch.exp(logps)
        print(type(topk))
        top_p, top_k = ps.topk(topk, dim = 1)

        #converting the output to numpy for later use 
        top_p, top_k = top_p.type(torch.FloatTensor).cpu().detach().numpy().reshape(-1), \
            top_k.type(torch.FloatTensor).cpu().detach().numpy().reshape(-1).astype(int).astype(str)

        return top_p, top_k

    
