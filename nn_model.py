import pandas as pd
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

# Create a PyTorch Embedding Class from SentenceTransformer

class Embedder(nn.Module):
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = "cpu"):
        super(Embedder, self).__init__()
        self.model = SentenceTransformer(model_name, device=device)

    def forward(self, x, batch_size: int):
        return self.model.encode(x, batch_size=batch_size, convert_to_tensor=True)
    
# 10 points

class FFNN(nn.Module):
    """
    A class representing our implementation of a Feed-Forward Neural Network.
    You will need to implement two methods:
        - A constructor to set up the architecture and hyperparameters of the model
        - The forward pass
    """
    
    def __init__(self, vocab_size: int, embedding_size: int, hidden_units=128, device: str = "cpu"):
        """
        Initialize a new untrained model. 
        
        You can change these parameters as you would like.
        Once you get a working model, you are encouraged to
        experiment with this constructor to improve performance.
        
        Params:
            vocab_size: The number of words in the vocabulary
            ngram: The value of N for training and prediction.
            embedding_layer: The previously trained embedder. 
            hidden_units: The size of the hidden layer.
        """        
        super().__init__()
        # YOUR CODE HERE
        # we recommend saving the parameters as instance variables
        # so you can access them later as needed
        # (in addition to anything else you need to do here)
        
		# Saving parameters as instance variables
        self.vocab_size = vocab_size
        #self.ngram = ngram
        self.hidden_units = hidden_units
        self.device = device

		# Save embedding size

        #embedding_size = embedding_layer.embedding_dim
        
		# Defining layers
        self.flatten = nn.Flatten() # Useful later to flatten array of ngram-1 after embedding before passing it to the linear layer
        self.linear_relu_stack = nn.Sequential(
			nn.Linear(in_features=embedding_size, out_features=hidden_units, bias=True),
			nn.ReLU(),
			nn.Linear(in_features=hidden_units, out_features=vocab_size, bias=True)
		)

        self.to(device)
        
    def forward(self, X: list) -> torch.tensor:
        """
        Compute the forward pass through the network.
        This is not a prediction, and it should not apply softmax.

        Params:
            X: the input data

        Returns:
            The output of the model; i.e. its predictions.
        
        """
        # YOUR CODE HERE
        flat_embedded = self.flatten(X)
        logits = self.linear_relu_stack(flat_embedded)
        return logits

# 10 points

# Defining a training function that goes over every batch per epoch
def train_one_epoch(dataloader, nn_model, optimizer, loss_fn):
    epoch_loss = 0

    for data in dataloader:
        # Separating the input + label pair for each instance
        inputs, labels = data
        inputs, labels = inputs.to(nn_model.device), labels.to(nn_model.device)

		# Zeroing gradients for every batch
        optimizer.zero_grad()
        
		# Make predictions for this batch
        outputs = nn_model(inputs)
        
		# Compute loss and gradients
        batch_loss = loss_fn(outputs, labels)
        batch_loss.backward()
        
		# Adjust learning weights
        optimizer.step()
        
		# Adding to epoch loss
        epoch_loss += batch_loss.item() # Covert scalar tensor into floating-point

    return epoch_loss

# Defining a general training function that goes over all the epochs
def train(dataloader, input_model, epochs: int = 1, lr: float = 0.001, 
          early_stop_threshold: int = None) -> None:
    """
    Our model's training loop with early stopping.
    Prints the cross entropy loss for each epoch and stops training early if
    the average loss falls below the specified threshold.

    Params:
        dataloader: The training dataloader
        input_model: The model we wish to train
        epochs: The maximum number of epochs to train for
        lr: Learning rate 
        early_stop_threshold: If set (a float value), training will stop early
                              when the average epoch loss falls below this threshold.
    """
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(input_model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()  # Applies log-softmax internally and computes the negative log likelihood
    
    n_batches = len(dataloader)
    
    # Ensure the model is in training mode.
    input_model.train()
    
    for epoch in tqdm(range(epochs)):
        epoch_loss = train_one_epoch(dataloader, input_model, optimizer, loss_fn)
        avg_epoch_loss = epoch_loss / n_batches
        print(f"Epoch: {epoch}, Loss: {avg_epoch_loss:.4f}\n")
        
        # Early stopping: if average loss falls below the threshold, break out of the loop.
        if early_stop_threshold is not None and avg_epoch_loss < early_stop_threshold:
            print(f"Early stopping triggered: Average loss {avg_epoch_loss:.4f} is below threshold {early_stop_threshold}.")
            break

#Function to preprocess the lyrics
def preprocess_sentence(lyrics:str,lemmatizer: WordNetLemmatizer = LEMMATIZER, 
                        stop_words: set = STOP_WORDS) -> str: 

         # Apply case-folding on your text.
        lyrics = lyrics.lower()

        # Remove any punctuations within your sentence.
        lyrics = lyrics.translate(str.maketrans('', '', string.punctuation))
        lyrics =lyrics.split('Lyrics', 1)[1].strip() if 'Lyrics' in lyrics else lyrics.strip()

        tokens = word_tokenize(lyrics)
  
        # Remove stop words and lemmatize your sentence if they are provided
        if stop_words is not None:
            tokens = [word for word in tokens if word not in stop_words]
        if lemmatizer is not None:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
        preprocessed = ' '.join(tokens)
        return preprocessed

def split_dataset(X, Y, device: str = "cpu"):
    """
    Splits the dataset X and corresponding labels Y into training and testing sets.
    For every block of 10 samples, it selects 8 for training and 2 for testing.
    
    If the total number of samples is not a multiple of 10, the remaining samples
    are split approximately in an 80/20 ratio between training and testing.
    
    Parameters:
        X (list or tensor-like): The feature data.
        Y (list or tensor-like): The corresponding labels.
        
    Returns:
        X_train_tensor (torch.Tensor): Training features.
        X_test_tensor  (torch.Tensor): Testing features.
        y_train_tensor (torch.Tensor): Training labels.
        y_test_tensor  (torch.Tensor): Testing labels.
    """
    # Check to ensure X and Y have the same length
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of elements.")
    
    X_train, X_test, y_train, y_test = [], [], [], []

    # Calculate number of complete 10-sample chunks
    num_full_chunks = len(X) // 10

    # Process each full chunk
    for i in range(0, num_full_chunks * 10, 10):
        # Select 8 for training and 2 for testing
        X_train.extend(X[i:i+8])
        X_test.extend(X[i+8:i+10])
        y_train.extend(Y[i:i+8])
        y_test.extend(Y[i+8:i+10])
    
    # Handle any remaining samples that don't form a complete chunk
    remainder = len(X) % 10
    if remainder > 0:
        # Determine count of training samples from the remainder (approximately 80%)
        train_count = int(remainder * 0.8)
        # In case the computed train_count is 0 (e.g. for a single remaining sample), ensure at least one sample goes to training.
        train_count = max(train_count, 1)
        
        # Add the remaining samples according to the split
        X_train.extend(X[-remainder:-remainder + train_count])
        X_test.extend(X[-remainder + train_count:])
        y_train.extend(Y[-remainder:-remainder + train_count])
        y_test.extend(Y[-remainder + train_count:])
    
    # Convert feature lists to tensors:
    # If the features are already PyTorch tensors, use torch.stack;
    # otherwise, we assume they are lists or numpy arrays and use torch.tensor.
    if X_train and isinstance(X_train[0], torch.Tensor):
        X_train_tensor = torch.stack(X_train).to(device)
    else:
        X_train_tensor = torch.tensor(X_train, device=device)

    if X_test and isinstance(X_test[0], torch.Tensor):
        X_test_tensor = torch.stack(X_test).to(device)
    else:
        X_test_tensor = torch.tensor(X_test, device=device)
    
    # Convert labels to tensors (assuming integer labels)
    y_train_tensor = torch.tensor(y_train, dtype=torch.int64, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.int64, device=device)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_dataloaders(X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,num_sequences_per_batch: int, 
                       shuffle: bool = True) -> tuple[torch.utils.data.DataLoader]:
    """
    Convert our data into a PyTorch DataLoader.    
    A DataLoader is an object that splits the dataset into batches for training.
    PyTorch docs: 
        https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
        https://pytorch.org/docs/stable/data.html

    Note that you have to first convert your data into a PyTorch DataSet.
    You DO NOT have to implement this yourself, instead you should use a TensorDataset.

    You are in charge of splitting the data into train and test sets based on the given
    test_pct. There are several functions you can use to acheive this!

    The shuffle parameter refers to shuffling the data *in the loader* (look at the docs),
    not whether or not to shuffle the data before splitting it into train and test sets.
    (don't shuffle before splitting)

    Params:
        X: A list of input sequences
        Y: A list of labels
        num_sequences_per_batch: Batch size
        shuffle: INSTRUCTORS ONLY

    Returns:
        One DataLoader for training, and one for testing.
    """

    # Validate that training and testing tensors have matching first dimensions.
    if X_train.size(0) != y_train.size(0):
        raise ValueError("X_train and y_train must have the same number of samples.")
    if X_test.size(0) != y_test.size(0):
        raise ValueError("X_test and y_test must have the same number of samples.")
    
    training_dataSet = TensorDataset(X_train, y_train)
    test_dataSet = TensorDataset(X_test, y_test)

    dataloader_train = DataLoader(training_dataSet, batch_size=num_sequences_per_batch, shuffle=shuffle)
    dataloader_test = DataLoader(test_dataSet, batch_size=num_sequences_per_batch, shuffle=shuffle)
    return dataloader_train, dataloader_test

def full_pipeline(x,y, vocab_size: int, 
                batch_size:int, hidden_units = 128, embedding_size: int = 384, epochs = 1,
                lr = 0.001, device: str = "cpu", early_stop_threshold: int = 1e-4
                ) -> FFNN:
    """
    Run the entire pipeline from loading embeddings to training.
    You won't use the test set for anything.

    Params:
        data: The raw data to train on, parsed as a list of lists of tokens
        word_embeddings_filename: The filename of the Word2Vec word embeddings
        batch_size: The batch size to use
        hidden_units: The number of hidden units to use
        epochs: The number of epochs to train for
        lr: The learning rate to use
        test_pct: The proportion of samples to use in the test set.

    Returns:
        The trained model.
    """
    # Loading embeddings

    x_train, x_test, y_train, y_test = split_dataset(x, y, device)

	# Create training dataloader
    dataloader_train, dataloader_test = create_dataloaders(x_train, x_test,y_train,y_test, batch_size)

	# Create FFNN model
    nn_model = FFNN(vocab_size=vocab_size, embedding_size=embedding_size, hidden_units=hidden_units, device=device)

	# Train our model
    train(dataloader=dataloader_train, input_model=nn_model, epochs=epochs, lr=lr, early_stop_threshold=early_stop_threshold)

    return nn_model, dataloader_test

def evaluate_model(model, dataloader) -> float:
    """
    Evaluate the model's accuracy on a dataset provided by a DataLoader.
    
    Parameters:
        model (nn.Module): The PyTorch model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader containing evaluation data.
        device (str, optional): The device to run inference on, e.g., "cpu" or "cuda".
            If None, the function will use model.device if available; otherwise, defaults to "cpu".
    
    Returns:
        float: Accuracy on the evaluation dataset.
    """
    
    # Ensure the model is in evaluation mode.
    model.eval()
    
    total_samples = 0
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Move inputs and labels to the specified device.
            #inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)

            # Instead of torch.max(...).data, we use torch.argmax for clarity.
            predicted = torch.argmax(outputs, dim=1)
            
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    print(f"Accuracy on test set: {accuracy * 100:.2f}% ({correct_predictions}/{total_samples})")
    return accuracy
