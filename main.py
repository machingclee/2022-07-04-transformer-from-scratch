from src.transformer import Transformer
from src.translator import Translator
from src.device import device
from src.dataset import data_loader, tgt_word_index, tgt_index_word, src_index_word
from src.train import train
import torch


def main():
    transformer = Transformer().to(device)
    model_path = "pths/model_epoch_30.pth"

    if model_path is not None:
        transformer.load_state_dict(torch.load(model_path))

    translator = Translator(transformer)
    enc_inputs, _, _ = next(iter(data_loader))
    enc_inputs = enc_inputs.to(device)
    # e.g. enc_inputs = tensor([
    #   [1, 2, 3, 4, 0], [1, 2, 3, 5, 0]
    # ], device='cuda:0')
    for i in range(len(enc_inputs)):
        enc_input = enc_inputs[i]
        sentence = " ".join([src_index_word[i.item()] for i in enc_input])
        print("source sentence:", sentence)
        predict = translator.translate_input_index(
            enc_input.unsqueeze(0),  # expand as batch
            src_start_index=tgt_word_index["<sos>"]
        )
        print(enc_input, '->', [tgt_index_word[n.item()]
                                    for n in predict.squeeze()])


if __name__ == "__main__":
    train(epochs=30, use_saved_vocab=True)
