import pickle
from torchtext.vocab import Vocab
from src.dataset_en_jp import Vocabs
from src.translator import Translator
from src.transformer import Transformer

def visualize(transformer:Transformer, enc_input, src_vocab:Vocab, tgt_vocab:Vocab):
    src_index_word = src_vocab.get_itos()
    tgt_index_word = tgt_vocab.get_itos()
    tgt_word_index = tgt_vocab.get_stoi()
    translator = Translator(transformer)
    sentence = " ".join([src_index_word[i.item()] for i in enc_input])
    print("source sentence:", sentence)
    predict = translator.translate_input_index(
        enc_input.unsqueeze(0),  # expand as batch
        src_start_index=tgt_word_index["<sos>"],
        tgt_word_index=tgt_word_index,
        tgt_index_word=tgt_index_word
    )
    
    try:
        print("--------------------")
        print("result", [tgt_index_word[n.item()]
                                for n in predict.squeeze()])
        print("--------------------")
    except Exception as err:
        print(f"{err}")