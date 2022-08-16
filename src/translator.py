
import torch
from src.device import device
from src.transformer import Transformer
from src.dataset import tgt_word_index, tgt_index_word


class Translator():
    def __init__(self, transformer: Transformer):
        self.transformer = transformer

    def translate(self, enc_input, start_index):
        dec_input = torch.zeros(1, 0).type_as(enc_input)
        terminated = False
        next_tgt_word_index = start_index
        while not terminated:
            dec_input = torch.cat(
                [
                    dec_input.detach(),
                    torch.tensor([[next_tgt_word_index]],dtype=enc_input.dtype).to(device)
                ],
                -1
            )
            dec_output_logits, _, _, _= self.transformer(enc_input, dec_input)
            next_tgt_word_index = torch.argmax(dec_output_logits[-1])

            if next_tgt_word_index == tgt_word_index["."]:
                terminated = True

            print("next_word", tgt_index_word[next_tgt_word_index.item()])
            
        # remove batch, remove <sos>
        return dec_input.squeeze(0)[1:]
