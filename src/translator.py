
import torch
from src.device import device
from src.transformer import Transformer
from src import config

class Translator():
    def __init__(self, transformer: Transformer):
        self.transformer = transformer

    def translate_input_index(self, enc_input, src_start_index, tgt_word_index, tgt_index_word):
        dec_input = torch.zeros(1, 0).type_as(enc_input)
        terminated = False
        next_tgt_word_index = src_start_index
        word_count = 0
        while not terminated:
            dec_input = torch.cat(
                [
                    dec_input.detach(),
                    torch.tensor([[next_tgt_word_index]],dtype=enc_input.dtype).to(device)
                ],
                -1
            )
            word_count += 1
            dec_output_logits, _, _, _= self.transformer(enc_input, dec_input)
            next_tgt_word_index = torch.argmax(dec_output_logits[-1])

            if next_tgt_word_index == tgt_word_index["<eos>"] or word_count == config.tgt_max_len + 1:
                terminated = True

        # remove batch, remove <sos>
        return dec_input.squeeze(0)[1:]
