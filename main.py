from src.transformer import Transformer
from src.translator import Translator
from src.device import device
from src.dataset import data_loader, tgt_word_index, tgt_index_word, src_index_word
from src.train import train
import torch


def main():
    train(
       epochs=30,
       use_saved_vocab=True,
       learning_rate=1e-3
    )


if __name__ == "__main__":
    main()
