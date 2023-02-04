import torch.nn as nn


class Attention(nn.Module):
    '''
    A class that implements the attention module that is to be included in the decoder
    for a show, attend and tell implementation.
    The current implementation is a deterministic soft attention model. This is smooth and differentiable.
    Thus, end to end learning is possible using backpropogation.

    Arguments:
        encoder_dim (int): The output dimension of the encoder. Default = 2048
    '''

    def __init__(self, encoder_dim):
        super(Attention, self).__init__()
        self.U = nn.Linear(512, 512)  # Linear layer to transform decoder's earlier output
        self.W = nn.Linear(encoder_dim, 512)  # Linear layer to transform the encoded image features
        self.v = nn.Linear(512, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)  # Computation of the weights - alpha

    def forward(self, img_features, hidden_state):
        '''
        Forward propogation through attention model

        Arguments:
            img_features (tensor): Extracted features from the encoder module.
                (beam_size, feature_pixels = 10x10, encoder_dim = 2048)
            hidden_state (tensor): Previous iteration decoder output. (beam_size, decoder_dim = 512)

        Output:
            weighted_img_features (tensor): Extracted features weighted based on current attention
                (beam_size, encoder_dim = 2048)
            alpha (tensor): Weights for the current attention (beam_size, feature_pixels= 10x10)
        '''
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(U_h + W_s)
        e = self.v(att).squeeze(2)
        alpha = self.softmax(e)
        weighted_img_features = (img_features * alpha.unsqueeze(2)).sum(1)
        return weighted_img_features, alpha
