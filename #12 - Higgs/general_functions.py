import pandas as pd
import numpy as np
#import numpy as numpy
import pickle
from keras.datasets import mnist
import matplotlib.pyplot as plt

# ______________________________________________________________________________
class NNPy:
    """
    Warm-up: numpy
    --------------

    A fully-connected ReLU network with one hidden layer and no biases, trained to
    predict y from x using Euclidean error.

    This implementation uses numpy to manually compute the forward pass, loss, and
    backward pass.

    A numpy array is a generic n-dimensional array; it does not know anything about
    deep learning or gradients or computational graphs, and is just a way to perform
    generic numeric computations.

    """
    import numpy as np
    @staticmethod
    def NeuralNet(self, input_data, output_labels, batch_size=64, input_dimension=1000, hidden_dimension=100, output_dimension=10, epochs=500):
        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        N, D_in, H, D_out = batch_size, input_dimension, hidden_dimension, output_dimension

        x, y = input_data, output_labels
        # Create random input and output data
        #x = np.random.randn(N, D_in)
        #y = np.random.randn(N, D_out)

        # Randomly initialize weights
        w1 = np.random.randn(D_in, H)
        w2 = np.random.randn(H, D_out)

        learning_rate = 1e-6
        for t in range(500):
            # Forward pass: compute predicted y
            h = x.dot(w1)
            h_relu = np.maximum(h, 0)
            y_pred = h_relu.dot(w2)

            # Compute and print loss
            loss = np.square(y_pred - y).sum()
            print(t, loss)

            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_y_pred = 2.0 * (y_pred - y)
            grad_w2 = h_relu.T.dot(grad_y_pred)
            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h = grad_h_relu.copy()
            grad_h[h < 0] = 0
            grad_w1 = x.T.dot(grad_h)

            # Update weights
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2

import numpy as np


    
# NN in Numpy from Kaggle ______________________________________________________
class CustomNN:
    """
    Chadocan on Kaggle   <--- love you my man, despite that you're       French
    """

    @staticmethod
    def ReLU(Z):
        #return np.max(Z,0)
        return np.maximum(Z,0)

    @staticmethod
    def derivative_ReLU(Z):
        return Z > 0

    @staticmethod
    def softmax(Z):
        """Compute softmax values for each sets of scores in x."""
        exp = np.exp(Z - np.max(Z)) #le np.max(Z) evite un overflow en diminuant le contenu de exp
        return exp / exp.sum(axis=0)

    @staticmethod
    def init_params(input_size, hidden_size=10, output_size=10):
        #W1 = np.random.randn(hidden_size, input_size)
        #b1 = np.random.randn(hidden_size, 1)
        #W2 = np.random.randn(output_size, hidden_size)
        #b2 = np.random.randn(output_size, 1)
        
        W1 = np.random.rand(hidden_size, input_size) - 0.5
        b1 = np.random.rand(hidden_size, 1) - 0.5
        W2 = np.random.rand(output_size, hidden_size) - 0.5
        b2 = np.random.rand(output_size, 1) - 0.5        
        
        return W1, b1, W2, b2

    @staticmethod
    def forward_propagation(X,W1,b1,W2,b2):
        Z1 = W1.dot(X) + b1 #10, m  # hidden_size x num_examples
        A1 = CustomNN.ReLU(Z1) # 10,m    # hidden_size x num_examples
        Z2 = W2.dot(A1) + b2 #10,m  # output_size x num_examples
        A2 = CustomNN.softmax(Z2) #10,m  # output_size x num_examples
        return Z1, A1, Z2, A2

    @staticmethod
    def one_hot(Y):
        ''' return an 0 vector with 1 only in the position correspondind to the value in Y'''
        one_hot_Y = np.zeros((Y.max()+1,Y.size)) #si le chiffre le plus grand dans Y est 9 ca fait 10 lignes
        one_hot_Y[Y,np.arange(Y.size)] = 1 # met un 1 en ligne Y[i] et en colonne i, change l'ordre mais pas le nombre
        return one_hot_Y

    @staticmethod
    def backward_propagation(X, Y, A1, A2, W2, Z1):
        m = Y.size
        one_hot_Y = CustomNN.one_hot(Y)
        dZ2 = 2*(A2 - one_hot_Y) #10,m # output_size x num_examples
        dW2 = 1/m * (dZ2.dot(A1.T)) # 10 , 10   # output_size x hidden_size
        db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True) # 10, 1  # output_size x 1
        dZ1 = W2.T.dot(dZ2) * CustomNN.derivative_ReLU(Z1) # 10, m   # hidden_size x num_examples
        dW1 = 1/m * (dZ1.dot(X.T)) #10, 784     # hidden_size x input_size
        db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True) # 10, 1  # hidden_size x 1

        return dW1, db1, dW2, db2

    @staticmethod
    def update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2):
        W1 -= alpha * dW1
        b1 -= alpha * np.reshape(db1, (10,1))
        #b1 -= alpha * db1
        W2 -= alpha * dW2
        b2 -= alpha * np.reshape(db2, (10,1))
        #b2 -= alpha * db2

        return W1, b1, W2, b2

    @staticmethod
    def get_predictions(A2):
        return np.argmax(A2, 0)

    @staticmethod
    def get_accuracy(predictions, Y):
        return np.sum(predictions == Y)/Y.size

    @staticmethod
    def gradient_descent(X, Y, alpha, iterations):
        size , Xm = X.shape

        W1, b1, W2, b2 = CustomNN.init_params(size)
        for i in range(iterations):
            Z1, A1, Z2, A2 = CustomNN.forward_propagation(X, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = CustomNN.backward_propagation(X, Y, A1, A2, W2, Z1)

            W1, b1, W2, b2 = CustomNN.update_params(alpha, W1, b1, W2, b2, dW1, db1, dW2, db2)   

            if (i+1) % int(iterations/10) == 0:
                print(f"Iteration: {i+1} / {iterations}")
                prediction = CustomNN.get_predictions(A2)
                print(f'{CustomNN.get_accuracy(prediction, Y):.3%}')
        return W1, b1, W2, b2

    @staticmethod
    def make_predictions(X, W1 ,b1, W2, b2):
        _, _, _, A2 = CustomNN.forward_propagation(X, W1, b1, W2, b2)
        predictions = CustomNN.get_predictions(A2)
        return predictions

    @staticmethod
    def show_prediction(index,X, Y, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR):
        # None => cree un nouvel axe de dimension 1, cela a pour effet de transposer X[:,index] qui un np.array de dimension 1 (ligne) et qui devient un vecteur (colonne)
        #  ce qui correspond bien a ce qui est demande par make_predictions qui attend une matrice dont les colonnes sont les pixels de l'image, la on donne une seule colonne
        vect_X = X[:, index,None]
        prediction = CustomNN.make_predictions(vect_X, W1, b1, W2, b2)
        label = Y[index]
        print("Prediction: ", prediction)
        print("Label: ", label)

        current_image = vect_X.reshape((WIDTH, HEIGHT)) * SCALE_FACTOR

        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.show()


# Importing and Plotting Higgs _________________________________________ChatGPT_
class ImportHiggs:
    @staticmethod
    def import_data(url, features):
        import tensorflow as tf
        gz = tf.keras.utils.get_file('HIGGS.csv.gz', url)
        ds = tf.data.experimental.CsvDataset(gz, [tf.float32] * (features + 1), compression_type="GZIP")
        return ds

    @staticmethod
    def pack_row(*row):
        import tensorflow as tf
        label = row[0]
        features = tf.stack(row[1:], axis=0)
        label = tf.reshape(label, [1])  # Reshape the label to [1]
        return features, label

    @staticmethod
    def preprocess_dataset(ds, features):
        packed_ds = ds.map(ImportHiggs.pack_row)
        return packed_ds

    @staticmethod
    def prepare_train_validate(packed_ds, n_validation, n_train, buffer_size, batch_size):
        validate_ds = packed_ds.take(n_validation).cache()
        train_ds = packed_ds.skip(n_validation).take(n_train).cache()
        validate_ds = validate_ds.batch(batch_size)
        train_ds = train_ds.shuffle(buffer_size).repeat().batch(batch_size)
        return train_ds, validate_ds

    @staticmethod
    def plot_histogram(packed_ds, batch_size):
        import matplotlib.pyplot as plt
        for features, label in packed_ds.batch(batch_size).take(1):
            plt.hist(features.numpy().flatten(), bins=101)
            plt.show()



print("Work ffs")

if __name__ == "__main__":
    #"""
    import matplotlib.pyplot as plt
    #from CustomNN import CustomNN

    # Constants
    FEATURES = 28
    URL = 'http://mlphysics.ics.uci.edu/data/higgs/HIGGS.csv.gz'
    N_VALIDATION = int(1e3)
    N_TRAIN = int(1e4)
    BUFFER_SIZE = int(1e4)
    BATCH_SIZE = 500

    # Import and preprocess the data
    ds = ImportHiggs.import_data(URL, FEATURES)
    packed_ds = ImportHiggs.preprocess_dataset(ds, FEATURES)
    ImportHiggs.plot_histogram(packed_ds, 1000)

    # Prepare training and validation datasets
    #train_ds, validate_ds = ImportHiggs.prepare_train_validate(packed_ds, N_VALIDATION, N_TRAIN, BUFFER_SIZE, BATCH_SIZE)

    # You can now use train_ds and validate_ds for training your model
    #"""

    import sys
    #sys.exit()
   
    ############## MAIN ##############

    import pandas as pd
    import numpy as np
    import pickle
    from keras.datasets import mnist
    import matplotlib.pyplot as plt

    #DIR = "C:\\Programming\\Mafijski Praktikum\\#12 - Higgs\\"
    #data = pd.read_csv(DIR+'train.csv\\train.csv')

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    SCALE_FACTOR = 255 # TRES IMPORTANT SINON OVERFLOW SUR EXP
    WIDTH = X_train.shape[1]
    HEIGHT = X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0],WIDTH*HEIGHT).T / SCALE_FACTOR
    X_test = X_test.reshape(X_test.shape[0],WIDTH*HEIGHT).T  / SCALE_FACTOR

    W1, b1, W2, b2 = CustomNN.gradient_descent(X_train, Y_train, 0.15, 1000)
    with open("trained_params.pkl","wb") as dump_file:
        pickle.dump((W1, b1, W2, b2),dump_file)

    with open("trained_params.pkl","rb") as dump_file:
        W1, b1, W2, b2=pickle.load(dump_file)
    CustomNN.show_prediction(0, X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
    CustomNN.show_prediction(1, X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
    CustomNN.show_prediction(2, X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
    CustomNN.show_prediction(100, X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
    CustomNN.show_prediction(200, X_test, Y_test, W1, b1, W2, b2, WIDTH, HEIGHT, SCALE_FACTOR)
