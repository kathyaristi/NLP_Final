# This was a model used with starter code for CS 4120, HW 3, Spring 2025

import numpy as np
import pickle

class LogisticRegression:
    """
    This class creates an instance of the Logistic Regression classifier.
    """
    def __init__(self, learning_rate: float, num_iterations: int) -> None:
        """
        Initialization function for the classifier.
        Args:
            learning_rate (float): Adjusts alpha to learn through gradient updates.
            num_iterations (int): Adjusts the number of iterations to train the dataset on.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Given a set of scores, normalize them to sum up to 1.
        Compare this with your implementation from HW 1.

        Args:
            x (np.ndarray): Individual scores corresponding to each class.

        Returns:
            np.ndarray: Probabilities corresponding to each class.
        """
        exp_scores = np.exp(x)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  

    def _init_weights(self, X: np.ndarray, y: list) -> None:
        """Initializes a weight matrix and biases to 0.

        Args:
            X (np.array): The sparse tfidf matrix.
            y (list): List of labels of each document.
        """
        vocab_length = len(np.unique(y))
        num_features = X.shape[1]
        
        self.weights = np.zeros((vocab_length, num_features))

        # Set up mapping for labels
        # single 1 at the index corresponding to the label
        self.label_mapping = {label: i for i, label in enumerate(np.unique(y))} # Set returns a random order every time

    
    # PROVIDED
    def _get_label_as_vector(self, y: list) -> np.ndarray:
        """
        "One hot" encodes an input list of labels. 
        This function takes a list of string labels and coverts each into
        a vector that has one 1 at the index corresponding to the label.
        The rest of the items in the vector are 0s.

        Args:
            y (list): List of labels of each document.

        Returns:
            list : a list of lists encoding the class for each document.
        """
        encoded = []
        for label in y:
            if label not in self.label_mapping:
                raise ValueError(f'Label {label} not found in training data.')
            else:
                one_hot = [0 if i != self.label_mapping[label] else 1 for i in range(len(self.label_mapping))]
                encoded.append(one_hot)
        return np.array(encoded)

    
    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        The cross entropy loss to compute distance between the golden labels and the predicted estimates.
        Calculates loss for multiple examples at once.

        Args:
            y_true (np.ndarray): One hot encoded vector indicating the true label. 
            y_pred (np.ndarray): Probability estimates corresponding to each class.

        Returns:
            float: Loss quantifying the distance between the gold labels and the estimates.
        """

        log_vals = -np.log(y_pred) * y_true
        return np.average(log_vals)
    
    def _load_model_weights(self, filename: str) -> None:
        with open(filename, 'rb') as file:
            self.weights = pickle.load(file)

        print(f"Weights loaded successfully from {filename}!")

    def train(self, X: np.ndarray, y: list, verbose: bool = False, load_weights_file = None) -> None:
        """
        Trains the model for a certain number of iterations. 
        Gradients must be computed and the weights/biases must be updated at the end of each iteration.
        You need not loop through individual documents - use matrix operations to compute losses instead.

        Hint: you'll need to be careful about matrix dimensions.
        To get the transpose of a matrix, you can use the .T attribute.
        e.g., X.T will return the transpose of X.
        
        Args:
            X (np.array): The sparse tfidf matrix as a numpy array.
            y (list): List of labels of each document.
            verbose (bool): If True, print the epoch number and the loss after each 100th iteration.
        """

        # add a 1 to the end of each row in the X matrix
        # this is to account for the bias term
        X = np.hstack((X, np.ones((X.shape[0], 1))))

        self._init_weights(X,y)

        encoded_y = self._get_label_as_vector(y)

        if verbose:
            print(f"Training for {self.num_iterations} iterations")
            print("class mappings: ", self.label_mapping)

        if load_weights_file is not None:
            self._load_model_weights(load_weights_file)
            return

        for i in range(self.num_iterations):
            # Compute the predictions for ALL documents
           
            dot_product = np.dot(X, self.weights.T)
            probabilities = self._softmax(dot_product)

            # Compute the losses and error

            losses = self._cross_entropy_loss(encoded_y, probabilities)
            error = probabilities - encoded_y

            # Print the loss after each 100th iteration if verbose is True
            if i%100 == 0 and verbose: 
              print(f"losses: {losses}")

            # Compute gradients and update weights/biases
            
            gradients = np.dot(error.T, X)/X.shape[0]

            self.weights = self.weights - np.multiply(self.learning_rate,gradients)
    
    def predict(self, X: np.ndarray, eval=False):
        """Create a function to return the genre a certain document vector belongs to.

        Args:
           X (np.array): The sparse tfidf vector for a single example to be labeled as a numpy array.

        Returns:
            str: A human readable class fetched from self.label_mapping
        """       
        # calculate the z scores for the document
        
        X = np.append(X,1)
        dot_product = np.dot(X, self.weights.T)

        z_score = np.reshape(dot_product, (1, -1))

        # then you can your z score array to the softmax function
        prediction = self._softmax(z_score)

        # then map the prediction to a class
        #returns index of max 
        predicted_label = np.argmax(prediction, axis=1)[0]

        # translate the labels back to human readable form

        for key,val in self.label_mapping.items():
          if val == predicted_label and eval:
            return key, np.max(prediction, axis=1)
          elif val == predicted_label:
            return key
      
        return -1

    def save_model_weights(self, filename_prefix='model/logistic_regression_weights'):
        weights = self.weights
        shape = weights.shape
        filename = f"{filename_prefix}_{shape[0]}x{shape[1]}.pkl"
        
        with open(filename, 'wb') as file:
            pickle.dump(weights, file)
            
        print(f"Weights saved successfully to {filename}!")

        

def save_predictions(true_labels: list, predictions: list, filename: str):
    """
    This function saves the predictions and true labels to a file.
    Args:
        true_labels (list): A list of true labels.
        predictions (list): A list of predicted labels.
        filename (str): The name of the file to save the predictions and true labels.
    """
    with open(filename, 'w') as f:
        f.write('true,prediction\n')
        for pred, true in zip(predictions, true_labels):
            f.write(f'{true},{pred}\n')