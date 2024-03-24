from torch import nn, optim, no_grad, load, save, float as torch_float, device as torch_device, zeros
from tqdm import tqdm


class NeuralNetwork(nn.Module):
    """
    Classe qui permet de créer un réseau de neurones
    en ajoutant des couches de neurones voulus, et
    qui permet l'ajout de couches de convolution.
    """
    def __init__(self, linear_relu_stack, conv_layer=None):
        """
        linear_relu_stack : Séquence de couches linéaires et de fonctions d'activation
        conv_layer : Séquence de couches de convolution (par défaut : None aucune couche de convolution)
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv_layer = conv_layer
        self.linear_relu_stack = linear_relu_stack

    def forward(self, x):
        """
        Fonction de propagation avant du réseau de neurones
        x : Tenseur d'entrée
        retourne les logits (sortie du réseau de neurones)
        """
        if self.conv_layer is not None:
            x = self.conv_layer(x)
            #print(x.shape)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
class Model:

    def __init__(self, model, loss_fn, device='cpu'):
        """
        Généralisation d'un modèle de réseau de neurones CNN pour l'apprentissage et le test
        model : modèle d'un réseau de neurones CNN
        loss_fn : fonction de perte
        device : appareil utilisé (par défaut : cpu)
        """
        self.model = model
        self.model.to(device)
        self.loss_fn = loss_fn
        self.device = device
    

    def learn(self, train_loader, nbr_epoch=10, learning_rate=0.1):
        """
        Fonction d'apprentissage du modèle
        train_loader : DataLoader de l'ensemble d'apprentissage
        nbr_epoch : nombre d'époques (par défaut : 10)
        learning_rate : taux d'apprentissage (par défaut : 0.1)
        Retourne la liste des pertes par batch
        """
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        loss_data = []
        for epoch in tqdm(range(nbr_epoch)):
            loss_data.append([])
            self.model.train() # Indique que le modèle est en mode apprentissage
            epoch_loss = 0
            for batch, (X, y) in tqdm(enumerate(train_loader)):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                #if batch%100 == 0:
                    #print(f"Epoch {epoch} : Batch: {batch} Loss {loss.item()}")
                epoch_loss += loss.item()
                loss_data[epoch].append(loss.item())
            print(f"Epoch {epoch} : Loss {epoch_loss/len(train_loader)}")
        return loss_data
    
    def test_model(self, test_loader):
        """
        Fonction de test du modèle
        test_loader : DataLoader de l'ensemble de test
        Retourne le taux de réussite et la perte moyenne
        """
        test_loss, correct = 0, 0
        with no_grad(): # Empêche le calcul du gradient
            for X, y in tqdm(test_loader): # X = image, y = label
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch_float).sum().item()
        test_loss /= len(test_loader.dataset)
        correct /= len(test_loader.dataset)
        print(f"Test Error : \n Accuracy : {(100*correct):>0.1f}%, Avg loss : {test_loss:>8f} \n")
        return correct, test_loss

    def save_model(self, path='models/character_models/model.pth'):
        """
        Procédure de sauvegarde du modèle
        path : chemin du fichier de sauvegarde (par défaut : models/character_models/model.pth)
        """
        save(self.model.state_dict(), path)
        print("Model saved")

    def load_model(self, path='models/character_models/model.pth'):
        """
        Procédure de chargement du modèle
        path : chemin du fichier de chargement (par défaut : models/character_models/model.pth)
        """
        self.model.load_state_dict(load(path, map_location=torch_device(self.device))) # map_location permet de charger le modèle sur l'appereil si on a entrainé le modèle sur le gpu
        #print("Model loaded")

    def predict(self, tensor):
        """
        Fonction de prédiction du modèle
        tensor : tenseur d'entrée
        Retourne les prédictions du modèle
        """
        self.model.eval()
        with no_grad():
            tensor = tensor.to(self.device)
            pred = self.model(tensor)
            pred = nn.functional.softmax(pred, dim=1)
            return pred

