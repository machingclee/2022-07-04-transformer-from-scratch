import pickle
import torch
from collections import Counter
from random import shuffle
# from torchtext.legacy.data import Field, BucketIterator, TabularDataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab, Vocab
from src.device import device
from src import config
from typing import Optional

src_max_length = config.src_max_len
tgt_max_lenth = config.tgt_max_len

class Vocabs:
    src_vocab = None
    tgt_vocab = None
    
    def __init__(self, src_vocab_pickle_path, tgt_vocab_pickle_path):
        self.src_vocab_pickle_path = src_vocab_pickle_path
        self.tgt_vocab_pickle_path = tgt_vocab_pickle_path
            
    def get_src_vocab(self) -> Vocab:
        if Vocabs.src_vocab is None:
            with open(self.src_vocab_pickle_path, 'rb') as handle:
                Vocabs.src_vocab = pickle.load(handle)
        
        return Vocabs.src_vocab
    
    def get_tgt_vocab(self):
        if Vocabs.tgt_vocab is None:
            with open(self.tgt_vocab_pickle_path, 'rb') as handle:
                Vocabs.tgt_vocab = pickle.load(handle)
        
        return Vocabs.tgt_vocab


class Corpus:   
    def __init__(
            self, 
            src_lang="en_core_web_sm", 
            tgt_lang="ja_core_news_sm", 
            delimiter="\t",
            src_vocab: Optional[Vocab] = None,
            tgt_vocab: Optional[Vocab] = None
        ):
        # the lang keys are used in defining "field object"
        # which is exactly the csv's header, the column name, the json key, etc.
        self.delimiter=delimiter
        
        self.src_tokenizer = get_tokenizer("spacy", language=src_lang)
        self.tgt_tokenizer = get_tokenizer("spacy", language=tgt_lang)
    
        src_counter = Counter()
        tgt_counter =  Counter()
        
        if src_vocab is not None and tgt_vocab is not None:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab
        else:
            for src_line in self.get_src_sentence_iter():
                src_counter.update(self.src_tokenizer(src_line))
            for tgt_line in self.get_tgt_sentence_iter():
                tgt_counter.update(self.tgt_tokenizer(tgt_line))
                
            # for label, line in 
            self.src_vocab = vocab(
                src_counter, 
                min_freq=2, 
                specials=('<ukn>', '<pad>')
            )
            self.tgt_vocab = vocab(
                tgt_counter, 
                min_freq=2, 
                specials=('<ukn>', '<sos>', '<eos>', '<pad>')
            )


                
        
    def get_src_sentence_iter(self):
        with open(config.data_path, encoding="utf-8") as f:
            for line in f:
                src_line, _ = line.split(self.delimiter)
                yield src_line
                
    def get_tgt_sentence_iter(self):
        with open(config.data_path, encoding="utf-8") as f:
            for line in f:
                _, tgt_line = line.split(self.delimiter)
                yield tgt_line

    def save_vocabs(self):
        vocabs = {
            "src": self.src_vocab,
            "tgt": self.tgt_vocab
        }
        for lang, vocab in vocabs.items():
            with open(f"{lang}.pickle", 'wb+') as handle:
                pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TransformerDataset(Dataset):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.src_sentences = list(self.corpus.get_src_sentence_iter())
        self.tgt_sentences = list(self.corpus.get_tgt_sentence_iter())

    def __getitem__(self, index):
        src_text = self.src_sentences[index]
        tgt_text = self.tgt_sentences[index]
        
        src_stoi = self.corpus.src_vocab.get_stoi()
        tgt_stoi =  self.corpus.tgt_vocab.get_stoi()
        
        src_tokens= self.corpus.src_tokenizer(src_text)
        tgt_tokens = self.corpus.tgt_tokenizer(tgt_text)
        
        src_pad_len = config.src_max_len - len(src_tokens)
        tgt_pad_len = config.tgt_max_len - len(tgt_tokens)
        
        if src_pad_len > 0:
            src_idxes = [src_stoi.get(token, src_stoi["<ukn>"]) for token in src_tokens] + [src_stoi["<pad>"]] * src_pad_len
        else: 
            src_idxes = [src_stoi.get(token, src_stoi["<ukn>"]) for token in src_tokens[:config.src_max_len]]
            
        if tgt_pad_len > 0:
            tgt_idxes = [tgt_stoi['<sos>']] + \
                        [tgt_stoi.get(token, src_stoi["<ukn>"]) for token in tgt_tokens] + \
                        [tgt_stoi['<eos>']] + \
                        [tgt_stoi["<pad>"]] * tgt_pad_len
        else:
            tgt_idxes = [tgt_stoi['<sos>']] + \
                        [tgt_stoi.get(token, src_stoi["<ukn>"]) for token in tgt_tokens[:config.tgt_max_len]] + \
                        [tgt_stoi['<eos>']] + \
                        [tgt_stoi["<pad>"]] * tgt_pad_len
                    
        return torch.as_tensor(src_idxes, device=device), torch.as_tensor(tgt_idxes, device=device)


    def __len__(self):
        return len(self.src_sentences)
    



if __name__ == "__main__":
    corpus = Corpus()
    corpus.save_vocabs()
    dataset = TransformerDataset(corpus)
    result = dataset[0]
    print(result)
    



    
    


