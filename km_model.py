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
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
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

# Function to see which songs are part of the clusters

def select_random_samples_by_cluster(X, labels, target_cluster, n_number):
    """
    Select n_number random samples from the dataset X whose predicted cluster label equals target_cluster.
    
    Parameters:
      - X: numpy.ndarray, the dataset where each row is a sample.
      - labels: numpy.ndarray, the predicted cluster labels for each sample in X.
      - target_cluster: int (or appropriate type), the specific cluster label to filter by.
      - n_number: int, the number of random samples to select from the target cluster.
    
    Returns:
      - A numpy.ndarray containing the randomly selected samples from X that belong to target_cluster.
    
    Raises:
      - ValueError: If no samples are found in the target cluster or if n_number exceeds the number of available samples.
    """
    # Get the indices of all samples belonging to the target cluster
    cluster_indices = np.where(labels == target_cluster)[0]
    
    # Check if there are any samples in the cluster
    if len(cluster_indices) == 0:
        raise ValueError(f"No samples found for cluster {target_cluster}")
    
    # Check if we have enough samples in the cluster to select n_number of them
    if n_number > len(cluster_indices):
        raise ValueError(f"Requested {n_number} samples, but only {len(cluster_indices)} samples are available in cluster {target_cluster}")
    
    # Randomly choose n_number indices from the filtered indices without replacement
    random_indices = np.random.choice(cluster_indices, size=n_number, replace=False)
    
    # Return the corresponding samples from X
    return X['song'].values[random_indices]

# Function to get the assigned clusters for a list of indices

def get_assigned_clusters(indices, labels):
    """
    Get the assigned clusters for a list of indices.

    Parameters:
    - indices: list of int, the indices of the data points.
    - labels: numpy.ndarray, the cluster labels for all data points.

    Returns:
    - list of int, the cluster labels for the specified indices.
    """
    return [labels[i] for i in indices]