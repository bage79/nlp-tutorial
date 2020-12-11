# %%
# code by Tae Hwan Jung(Jeff Jung) @graykode, Derek Miller @dmmiller612
# Reference : https://github.com/jadore801120/attention-is-all-you-need-pytorch
#           https://github.com/JayParks/transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

# S: Symbol that shows the start of decoding input
# E: Symbol that shows the end of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

def make_batch(sentences):
    """ return vocab ids """
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k
    return pad_attn_mask

def get_attn_subsequent_mask(seq):  # seq=dec_input
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]    # attn_shape = [B, S, S]

    # all_ones:
    #   [[[1., 1., 1., 1., 1.],
    #     [1., 1., 1., 1., 1.],
    #     [1., 1., 1., 1., 1.],
    #     [1., 1., 1., 1., 1.],
    #     [1., 1., 1., 1., 1.]]])
    all_ones = np.ones(attn_shape)

    # subsequent_mask:
    #   [[[0., 1., 1., 1., 1.],
    #     [0., 0., 1., 1., 1.],
    #     [0., 0., 0., 1., 1.],
    #     [0., 0., 0., 0., 1.],
    #     [0., 0., 0., 0., 0.]]])
    subsequent_mask = np.triu(all_ones, k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).byte()
    return subsequent_mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q, K, V: (B, H, S, W) = (1, 8, 5, 64)

        # scores: (B, H, S, W) * (B, H, W, S) = (B, H, S, S) = (1, 8, 5, 5)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.

        # attn: (B, H, S, S) = (1, 8, 5, 5)
        attn = nn.Softmax(dim=-1)(scores)

        # context: (B, H, S, S) * (B, H, S, W) = (B, H, S, W)
        context = torch.matmul(attn, V) # (1, 8, 5, 64)
        return context, attn    # (B, H, S, W), (B, H, S, S)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask): # Q,K,V: (B, S, D), attn_mask: (B, S, S)
        # B=batch_size=1, S=len_q=len_k=5, D=d_model=512, H=n_heads=8, W=d_k=d_q=d_v=64
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W) = (1, 8, 5, 64)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        # (B, S, S) unsqueeze-> (B, 1, S, S) repeat-> (B, H, S, S) = (1, 8, 5, 5)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: (B, H, S, W)
        # attn: (B, H, S, S)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # context: (B, H, S, W) transpose-> (B, S, H, W) view-> (B, S, H * W)=(B, S, D)=(1, 5, 512)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v]

        # output: (B, S, D) -> (B, S, D)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn # output: (B, S, D), attn: (B, H, S, S)

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        # in_channels=512, out_channels=2048
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):  # (B, S, D) -> (B, S, D)
        # inputs: (B, S, D)
        residual = inputs

        # output: (B, S, D) transpose-> (B, in_channels=D, S) conv1d-> (B, out_channels=d_ff, S)=(1, 2048, 5)
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))

        # output: (B, in_channels=d_ff, S) conv1d-> (B, out_channels=D, S) transpose-> (B, S, D)=(1, 5, 512)
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):  # enc_inputs:(B, S, D), enc_self_attn_mask: (B, S, S)
        # enc_outputs: (B, S, D)
        # attn: (B, H, S, S)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn # enc_outputs: (B, S, D), ->attn: (B, H, S, S)

class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model),freeze=True)
        # self.pos_emb =
        # tensor([[ 0.0000e+00,  1.0000e+00,  0.0000e+00,  ...,  1.0000e+00,
        #   0.0000e+00,  1.0000e+00],
        # [ 8.4147e-01,  5.4030e-01,  8.2186e-01,  ...,  1.0000e+00,
        #   1.0366e-04,  1.0000e+00],
        # [ 9.0930e-01, -4.1615e-01,  9.3641e-01,  ...,  1.0000e+00,
        #   2.0733e-04,  1.0000e+00],
        # [ 1.4112e-01, -9.8999e-01,  2.4509e-01,  ...,  1.0000e+00,
        #   3.1099e-04,  1.0000e+00],
        # [-7.5680e-01, -6.5364e-01, -6.5717e-01,  ...,  1.0000e+00,
        #   4.1465e-04,  1.0000e+00],
        # [-9.5892e-01,  2.8366e-01, -9.9385e-01,  ...,  1.0000e+00,
        #   5.1832e-04,  1.0000e+00]])
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs): # enc_inputs: (B, S) -> enc_outputs: (B, S, D), enc_self_attns: (n_layers, (B, H, S, S))
        src_emb = self.src_emb(enc_inputs)
        pos_emb = self.pos_emb(enc_inputs.long()) # self.pos_emb(torch.LongTensor([[1, 2, 3, 4, 0]]))
        enc_outputs = src_emb + pos_emb
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # True=pad
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model),freeze=True)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs): # dec_inputs : [batch_size x target_len]
        dec_outputs = self.tgt_emb(dec_inputs) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)  # True=pad
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)    # upper triangle (1, 0)
        # dec_self_attn_mask:
        #[[[False,  True,  True,  True,  True],
        #  [False, False,  True,  True,  True],
        #  [False, False, False,  True,  True],
        #  [False, False, False, False,  True],
        #  [False, False, False, False, False]]])
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        # dec_enc_attn_mask:
        #[[[False, False, False, False,  True],
        #  [False, False, False, False,  True],
        #  [False, False, False, False,  True],
        #  [False, False, False, False,  True],
        #  [False, False, False, False,  True]]])
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)   # with enc_inputs

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # D -> tgt_vocab_size
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_outputs: (B, S, D)
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        # dec_outputs: (B, S, D)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        # dec_logits : [B, S, tgt_vocab_size]=(1, 5, 7)
        dec_logits = self.projection(dec_outputs)

        _tgt_vocab_size = dec_logits.size(-1)    # 7

        # output: (S, vocab_size)=(src_len, tgt_vocab_size)
        output = dec_logits.view(-1, _tgt_vocab_size)
        return output, enc_self_attns, dec_self_attns, dec_enc_attns

def showgraph(attn):
    attn = attn[-1].squeeze(0)[0]
    attn = attn.squeeze(0).data.numpy()
    fig = plt.figure(figsize=(n_heads, n_heads)) # [n_heads, n_heads]
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attn, cmap='viridis')
    ax.set_xticklabels(['']+sentences[0].split(), fontdict={'fontsize': 14}, rotation=90)
    ax.set_yticklabels(['']+sentences[2].split(), fontdict={'fontsize': 14})
    plt.show()

if __name__ == '__main__':
    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}    # 'P' = pad
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    number_dict = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5 # length of source
    tgt_len = 5 # length of target

    d_model = 512  # Embedding Size
    d_ff = 2048  # hidden layer dimension of PoswiseFeedForwardNet
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    enc_inputs, dec_inputs, target_batch = make_batch(sentences)
    enc_self_attns, dec_self_attns, dec_enc_attns = None, None, None
    for epoch in range(20):
        optimizer.zero_grad()
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)

        # outputs: (5, 7)
        loss = criterion(outputs, target_batch.contiguous().view(-1))

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()

    # Test
    predict, _, _, _ = model(enc_inputs, dec_inputs)
    predict = predict.data.max(1, keepdim=True)[1]
    print(sentences[0], '->', [number_dict[n.item()] for n in predict.squeeze()])

    print('first head of last state enc_self_attns')
    showgraph(enc_self_attns)

    print('first head of last state dec_self_attns')
    showgraph(dec_self_attns)

    print('first head of last state dec_enc_attns')
    showgraph(dec_enc_attns)