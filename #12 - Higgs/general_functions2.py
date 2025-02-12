import numpy as np
import matplotlib.pyplot as plt

class CustomNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        #self.b1 = np.zeros((hidden_size, 1))
        self.b1 = np.random.randn(hidden_size, 1) * 0.01
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        #self.b2 = np.zeros((output_size, 1))
        self.b2 = np.random.randn(output_size, 1) * 0.01

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def compute_loss(self, A2, Y):
        #m = Y.shape[1]
        m = Y.shape[0]
        log_probs = -np.log(A2[Y, range(m)])
        loss = np.sum(log_probs) / m
        return loss

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dA1 = np.dot(self.W2.T, dZ2)
        dZ1 = dA1 * (Z1 > 0)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def one_hot(self, Y):
        ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
        one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
        one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
        return one_hot_Y

    @staticmethod
    def compute_auc(fpr, tpr):
        return np.trapz(tpr, fpr)
    
    @staticmethod
    def compute_roc_auc(Y_true, Y_probs):
        thresholds = np.sort(Y_probs)[::-1]
        tps = []
        fps = []
        tpr = []
        fpr = []

        P = np.sum(Y_true == 1)
        N = np.sum(Y_true == 0)

        for threshold in thresholds:
            TP = np.sum((Y_probs >= threshold) & (Y_true == 1))
            FP = np.sum((Y_probs >= threshold) & (Y_true == 0))

            TPR = TP / P
            FPR = FP / N

            tps.append(TP)
            fps.append(FP)
            tpr.append(TPR)
            fpr.append(FPR)

        fps, tps = np.array(fps), np.array(tps)
        fpr, tpr = np.array(fpr), np.array(tpr)
        
        return [fps, tps, fpr, tpr, thresholds]


    def train(self, X, Y, epochs, learning_rate, device="cpu", get_ROC_AUC="custom"):
        Y_one_hot = np.eye(10)[Y].T
        #Y_one_hot = self.one_hot(Y)
        roc_aucs = []
        accuracies = []
        losses = []

        for epoch in range(epochs):
            Z1, A1, Z2, A2 = self.forward(X)
            loss = self.compute_loss(A2, Y)
            dW1, db1, dW2, db2 = self.backward(X, Y_one_hot, Z1, A1, Z2, A2)
            self.update_params(dW1, db1, dW2, db2, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
            
            if (epoch+1) % int(epochs/10) == 0:
                print(f"Iteration: {epoch+1} / {epochs}")
                #prediction = self.predict(A2, X)
                prediction = self.predict(X)
                accuracy = self.get_accuracy(prediction, Y)
                print(f'{accuracy:.3%}')
                accuracies.append(accuracy)
                losses.append(loss)

                if (get_ROC_AUC == "custom"):
                    # Calculate ROC and AUC manually
                    values = self.compute_roc_auc(Y_one_hot[1], A2[1])
                    auc = self.compute_auc(values[2], values[3])
                #elif (get_ROC_AUC == "fast"):
                #    from sklearn.metrics import roc_auc_score, roc_curve
                #    fpr, tpr, thresholds = roc_curve(Y_one_hot.T, A2.T)#, multi_class='ovr')
                #    auc = roc_auc_score(fpr, tpr)
                roc_aucs.append([auc]+values)

        
        weights = [self.W1, self.b1, self.W2, self.b2]
        return weights, roc_aucs, accuracies, losses

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=0)
    
    def predict_proba(self, X):
        _, _, _, A2 = self.forward(X)
        return A2
    
    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y)/Y.size
    
    def show_prediction(self, index, X, Y, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR):
        # None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
        #  ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice dont les colonnes sont les pixels de l'image, la on donne une seule colonne
        vect_X = X[:, index,None]
        prediction = self.predict(vect_X)
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()



import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import GPUtil

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

class TrainPlot:
    def evaluate_model(model, X_test, y_test, batch_size, device):
        if isinstance(model, torch.nn.Module):
            model.eval()
            correct = 0
            total = 0
            loss = 0.0
            criterion = torch.nn.CrossEntropyLoss()
            with torch.no_grad():
                for i in range(0, len(X_test), batch_size):
                    inputs = torch.tensor(X_test[i:i+batch_size], dtype=torch.float).to(device)
                    labels = torch.tensor(y_test[i:i+batch_size], dtype=torch.long).to(device)
                    outputs = model(inputs)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            loss /= total
        elif isinstance(model, tf.keras.Model):
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

        # 
        elif isinstance(model, MLPClassifier):
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            fpr, tpr = roc_curve(y_test, predictions)
            auc = roc_auc_score(y_test, predictions)
            loss = 0  # Implement loss calculation as needed

        elif isinstance(model, CustomNN):
            predictions = model.predict(X_test)

        return accuracy, loss


    def train_model(model, train_func, X_train, y_train, X_test, y_test, epochs, batch_size, use_gpu=False):
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        cpu_usage = []
        gpu_usage = []
        memory_usage = []
        accuracies = []
        losses = []

        start_time = time.time()

        #for epoch in range(epochs):
        epoch_start = time.time()
        
        # CPU & Memory Usage Monitoring
        cpu_before = psutil.cpu_percent(interval=None)
        mem_before = psutil.virtual_memory().percent

        # GPU Usage Monitoring (using GPUtil)
        gpu_before = 0
        if use_gpu:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_before = gpus[0].load * 100  # GPU load in percentage

        # Training Step
        if isinstance(model, CustomNN):
            weights, roc_aucs, accuracies, losses = train_func(X_train, y_train, epochs=epochs, learning_rate=0.01, get_ROC_AUC="custom")
        else:
            train_func(model, X_train, y_train, batch_size, device)
            # Evaluation
            accuracy, loss = evaluate_model(model, X_test, y_test, batch_size, device)
        
        # CPU & Memory Usage Monitoring after training step
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = psutil.virtual_memory().percent
        
        # GPU Usage Monitoring after training step
        gpu_after = 0
        if use_gpu:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_after = gpus[0].load * 100  # GPU load in percentage

        # Log Usage
        cpu_usage.append(cpu_after - cpu_before)
        gpu_usage.append(gpu_after - gpu_before)
        memory_usage.append(mem_after - mem_before)
        accuracies.append(accuracy)
        losses.append(loss)
        
        #print(f'Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {time.time() - epoch_start:.2f}s')

        total_time = time.time() - start_time

        return total_time, cpu_usage, gpu_usage, memory_usage, accuracies, losses
