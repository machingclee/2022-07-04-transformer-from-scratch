d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
d_k = d_v = d_model//n_heads # dimension of K(=Q), V