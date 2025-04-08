
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import torch
from gensim.models import Word2Vec
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words('english'))

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


def train_word2vec(data: list[list[str]], embeddings_size: int,
                    window: int = 5, min_count: int = 1, sg: int = 1) -> Word2Vec:
    """
    Create new word embeddings based on our data.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A gensim Word2Vec model
        https://radimrehurek.com/gensim/models/word2vec.html

    """

    #train using gensim
    model = Word2Vec(data, vector_size=embeddings_size, window=window, min_count=min_count, sg=sg)
    return model


def create_embedder(raw_embeddings: Word2Vec) -> torch.nn.Embedding:
    """
    Create a PyTorch embedding layer based on our data.

    We will *first* train a Word2Vec model on our data.
    Then, we'll use these weights to create a PyTorch embedding layer.
        `nn.Embedding.from_pretrained(weights)`


    PyTorch docs: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding.from_pretrained
    Gensim Word2Vec docs: https://radimrehurek.com/gensim/models/word2vec.html

    Pay particular attention to the *types* of the weights and the types required by PyTorch.

    Params:
        data: The corpus
        embeddings_size: The dimensions in each embedding

    Returns:
        A PyTorch embedding layer
    """

    # Hint:
    # For later tasks, we'll need two mappings: One from token to index, and one from index to tokens.
    # It might be a good idea to store these as properties of your embedder.
    # e.g. `embedder.token_to_index = ...`

    # Create mappings
    
    #get word vectors
    word_vectors = raw_embeddings.wv.vectors  
    #convert to tensor (weights of correct size)
    wv_tensor = torch.tensor(word_vectors, dtype=torch.float32)
    #pass in new weights  
    embedding = torch.nn.Embedding.from_pretrained(wv_tensor)

    token_to_index = dict()
    index_to_token = dict()
    for token in raw_embeddings.wv.index_to_key:
        token_to_index[token] = raw_embeddings.wv.key_to_index[token]
        index_to_token[raw_embeddings.wv.key_to_index[token]] = token

    embedding.token_to_index = token_to_index
    embedding.index_to_token = index_to_token
    #return embedding
    return embedding


def save_word2vec(embeddings: Word2Vec, filename: str) -> None:
    """
    Saves weights of trained gensim Word2Vec model to a file.

    Params:
        obj: The object.
        filename: The destination file.
    """
    embeddings.save(filename)

# PROVIDED
def load_word2vec(filename: str) -> Word2Vec:
    """
    Loads weights of trained gensim Word2Vec model from a file.

    Params:
        filename: The saved model file.
    """
    return Word2Vec.load(filename)

def split_dataset(X,Y):
    X_train, X_test, y_train, y_test = [], [], [], []

    for i in range(0, len(X), 10):
        X_train_chunk = X[i:i+8]  # Select 8 samples for training
        X_test_chunk = X[i+8:i+10]  # Select 2 sample for testing
        y_train_chunk = Y[i:i+8]  # Corresponding labels for training
        y_test_chunk = Y[i+8:i+10]  # Corresponding label for testing
    
        # Append chunks to total lists
        X_train.extend(X_train_chunk)        
        X_test.extend(X_test_chunk)
        y_train.extend(y_train_chunk)
        y_test.extend(y_test_chunk)

    X_train_tensor = torch.stack(X_train) if isinstance(X_train[0], torch.Tensor) else torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.stack(X_test) if isinstance(X_test[0], torch.Tensor) else torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def create_dataloaders(X_train: torch.Tensor, X_test: torch.Tensor, y_train: torch.Tensor, y_test: torch.Tensor,num_sequences_per_batch: int, 
                       test_pct: float = 0.1, shuffle: bool = True) -> tuple[torch.utils.data.DataLoader]:
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
        test_pct: The proportion of samples to use in the test set.
        shuffle: INSTRUCTORS ONLY

    Returns:
        One DataLoader for training, and one for testing.
    """
    #X_tensor = torch.tensor(X, dtype=torch.float32)  # X is the embeddings from SentenceTransformer

    training_dataSet = TensorDataset(X_train, y_train)
    test_dataSet = TensorDataset(X_test, y_test)

    
   # test_size = int(len(dataSet)*test_pct)
   #train_size = len(dataSet) - test_size
    #train_data, test_data = torch.utils.data.random_split(dataSet, [train_size, test_size])
    dataloader_train = DataLoader(training_dataSet, batch_size=num_sequences_per_batch, shuffle=shuffle)
    dataloader_test = DataLoader(test_dataSet, batch_size=num_sequences_per_batch, shuffle=shuffle)
    return dataloader_train, dataloader_test
#X_train = np.vstack(X_train)
#X_test = np.vstack(X_test)
#y_train = np.hstack(y_train)
#y_test = np.hstack(y_test)