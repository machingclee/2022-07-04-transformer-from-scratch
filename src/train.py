import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from src.transformer import Transformer
from src.device import device
# from torchsummary import summary
from torch import optim
from src.dataset_en_jp import Corpus
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.dataset_en_jp import TransformerDataset, Vocabs
from src import config
from src.console_log import ConsoleLog
from src.visualize import visualize

console_log = ConsoleLog(lines_up_on_end=1)

def train(epochs=10, use_saved_vocab=False):
    if use_saved_vocab:
        vocabs = Vocabs(src_vocab_pickle_path="src.pickle", tgt_vocab_pickle_path="tgt.pickle")
        corpus = Corpus(src_vocab=vocabs.get_src_vocab(), tgt_vocab=vocabs.get_tgt_vocab())
    else:
        corpus = Corpus()
        corpus.save_vocabs()
        
    src_vocab_size = len(corpus.src_vocab.get_stoi())
    tgt_vocab_size = len(corpus.tgt_vocab.get_stoi())
    
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size  
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.SGD(transformer.parameters(), lr=1e-4, momentum=0.99)
    dataset = TransformerDataset(corpus)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=config.batch_size,
                             shuffle=True
                             )
    # when arrived this step, pickle file must have been saved
    vocabs = Vocabs(src_vocab_pickle_path="src.pickle", tgt_vocab_pickle_path="tgt.pickle")
    src_vocabs = vocabs.get_src_vocab()
    tgt_vocabs = vocabs.get_tgt_vocab()
    
    for epoch in range(epochs):
        
        for batch_id, (src_idxes, tgt_idxes) in enumerate(tqdm(data_loader)):
            batch_id += 1            
            enc_inputs = src_idxes.to(device)
            dec_inputs = tgt_idxes[:, :-1].to(device)
            dec_outputs = tgt_idxes[:, 1:].to(device)

            outputs, _, _, _ = transformer(
                enc_inputs,
                dec_inputs
            )
            
            loss = criterion(outputs, dec_outputs.flatten())
            
            with torch.no_grad():
                console_log.print([
                    ("loss", loss.item())
                ])
                if batch_id % 10 == 0:
                    visualize(transformer, 
                              enc_inputs[0], 
                              src_vocabs, 
                              tgt_vocabs
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state_dict = transformer.state_dict()
        torch.save(state_dict, os.path.join("pths", f"model_epoch_{epoch}.pth"))
