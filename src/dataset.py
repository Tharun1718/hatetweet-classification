import torch
from torch.utils.data import Dataset
from collections import Counter
from .utils import clean_tweet

class Vocabulary:
    """
    Builds a mapping of words to integers (indices).
    """
    def __init__(self, freq_threshold=2):
        # Specific tokens for padding and unknown words
        self.itos = {0: "<PAD>", 1: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2  # Start index after PAD and UNK
        
        for sentence in sentence_list:
            for word in sentence.split():
                frequencies[word] += 1
                
                # Only add words that appear enough times
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """Converts a text string into a list of integers."""
        tokenized_text = clean_tweet(text).split()
        
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class HateSpeechDataset(Dataset):
    def __init__(self, df, vocab=None, is_test=False):
        self.df = df
        self.tweets = df['tweet']
        self.is_test = is_test
        self.vocab = vocab

        if not self.is_test:
            self.labels = df['label']
        else:
            self.labels = None
        
        # If no vocab provided, build it from this data
        if self.vocab is None:
            self.vocab = Vocabulary(freq_threshold=2)
            cleaned_texts = [clean_tweet(txt) for txt in self.tweets]
            self.vocab.build_vocabulary(cleaned_texts)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet_text = self.tweets.iloc[index]
        numericalized_text = self.vocab.numericalize(tweet_text)
        
        # Convert to Tensor
        text_tensor = torch.tensor(numericalized_text, dtype=torch.long)
        
        if self.is_test:
            return text_tensor
        else:
            label = self.labels.iloc[index]
            return text_tensor, torch.tensor(label, dtype=torch.float)
        

class CollateFn:
    """
    Custom collate function to handle batches with variable length sentences.
    It pads them to the length of the longest sentence in the batch.
    """
    def __init__(self, pad_idx, is_test=False):
        self.pad_idx = pad_idx
        self.is_test = is_test

    def __call__(self, batch):
        if self.is_test:
            # Batch is just a list of text_tensors
            texts = batch 
            texts = torch.nn.utils.rnn.pad_sequence(
                texts, batch_first=True, padding_value=self.pad_idx
            )
            return texts
        else:
            # Batch is a list of (text_tensor, label) tuples
            texts = [item[0] for item in batch]
            labels = [item[1] for item in batch]
            
            # Pad sequences
            texts = torch.nn.utils.rnn.pad_sequence(
                texts, batch_first=True, padding_value=self.pad_idx
            )
            
            # Stack labels
            labels = torch.stack(labels)
        
        return texts, labels