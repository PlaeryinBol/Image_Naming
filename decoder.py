from itertools import chain
import random
import torch
import torch.nn as nn
from attention import Attention
import torch.nn.functional as F
import config


class Decoder(nn.Module):
    '''
    A class that implements the decoder module with attention for the show, attend and tell implementation.
    The decoder is a recurrent neural network and the main unit in it is the LSTMCell.
    As we need to insert the attention component, the network is created using LSTM.

    This decoder implements Scheduled Sampling for the implementation of Teacher Forcing.
    This means, based on the provided teacher forcing ratio,
    teacher forcing will only be utilized for a random set of batches during training while the remaining will train
    without teacher forcing. This is done to attain a tradeoff between the pros and cons of using teacher forcing.

    Arguments:
        device (str): 'cuda' or 'cpu' based on which device is available
        vocabulary_size (int): Number of words in the word_dict (vocabulary of the network). Default = len(word_dict)
        encoder_dim (int): The output dimension of the encoder. Default = 2048
        tf_ratio (float): Teacher forcing ratio (must lie between 0 and 1). Default = 0
                          tf_ratio = 0 --> Teacher forcing will be always used
                          tf_ratio = 1 --> Teacher forcing will never be used
    '''

    def __init__(self, device, vocabulary_size, encoder_dim, tf_ratio=0):
        super(Decoder, self).__init__()
        self.tf_ratio = tf_ratio
        self.device = device
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, 512)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, 512)  # linear layer to find initial cell state of LSTMCell
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(512, encoder_dim)  # Linear layer to create a sigmoid activated gate
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(512, vocabulary_size)  # linear layer to find scores over the entire vocabulary
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim)  # attention module
        self.embedding = nn.Embedding(vocabulary_size, 512)  # embedding layer
        self.lstm = nn.LSTM(512 + encoder_dim, 512, 1)

    def forward(self, img_features, captions):
        """
        Forward propogation through the decoder network.

        Arguments:
            img_features (tensor): Extracted features from the encoder module.
                (batch_size, feature_pixels = 10x10, encoder_dim = 2048)
            captions (tensor): Captions encoded as keys of the word dictionary. (batch_size, max_caption_length)

        Output:
            preds (tensor): Prediction scores over the entire vocabulary
            alphas (tensor): Weights for the current attention
        """

        batch_size = img_features.size(0)

        # Get the initial LSTM state
        h, c = self.get_init_lstm_state(img_features)

        # As we are using a single LSTMCell, in order to generate the entire caption
        # we need to iterate maximum caption length number of times to generate the complete predicted caption
        max_timespan = max([len(caption) for caption in captions]) - 1

        # Determine whether teacher forcing is to be used for the current batch or not
        # based on the provided teacher forcing ratio
        batch_tf = True if random.random() > self.tf_ratio else False

        prev_words = torch.zeros(batch_size, 1).long().to(self.device)
        # If teacher forcing is to be used, then the ideal output is provided to the network,
        # otherwise the previous output of the network is given as input
        if batch_tf and self.training:
            embedding = self.embedding(captions)
        else:
            embedding = self.embedding(prev_words)

        # Create tensors to hold prediction scores and alpha - weights
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(self.device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(self.device)

        # At each time-step, decode by attention-weighing the encoder's outputbased on the decoder's previous hidden
        # state output then generate a new word in decoder with previous word and attention weighted encoding
        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            if batch_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))[1]
            h, c = h.squeeze(0), c.squeeze(0)
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training or not batch_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        '''
        Function to get the initial hidden state and cell state of LSTM based on encoded images.

        Arguments:
            img_features (tensor): Extracted features from the encoder module.
                (batch_size, feature_pixels = 14x14, encoder_dim = 512)
        '''
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c

    def caption(self, img_features, beam_size):
        """
        Function to generate the caption for the corresponding encoded image
        using beam search to provide the most optimal caption combination. This function is useful during
        human evaluation of the decoder to assess the quality of produced captions and while
        producing visualizations of attention and corresponding produced words.

        Arguments:
            img_features (tensor): Extracted features from the encoder module.
                (batch_size, feature_pixels = 14x14, encoder_dim = 512)
            beam_size (int): Number of top candidates to consider for beam search. Default = 3

        Output:
            sentence (list): ordered list of words of the final optimal caption
            alpha (tensor): weights corresponding to the generated caption
        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))[1]
            h, c = h.squeeze(0), c.squeeze(0)
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output
            output = F.log_softmax(output, dim=1)
            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)
            prev_word_idxs = prev_word_idxs.long()

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha

    def get_scores(self, k_prev_words, img_features, h, c, top_k_scores):
        """
        Function to get scores with LSTM.

        Arguments:
            k_prev_words (tensor): Previos words.
            img_features (tensor): Extracted features from the encoder module.
            h (tensor): Hidden value.
            c (tensor): Cell state.
            top_k_scores (tensor): Word scores of sentence.

        Output:
            h (tensor): Hidden value.
            c (tensor): Cell state.
            top_k_scores (tensor): Word scores of sentence.
        """
        embedding = self.embedding(k_prev_words).squeeze(1)
        context = self.attention(img_features, h)[0]
        gate = self.sigmoid(self.f_beta(h))
        gated_context = gate * context
        input_seq = torch.cat([embedding, gated_context], dim=1)
        h, c = self.lstm(input_seq.unsqueeze(0), (h.unsqueeze(0), c.unsqueeze(0)))[1]
        scores = self.deep_output(h.squeeze(0))
        scores = top_k_scores.expand_as(scores) + scores
        return h.squeeze(0), c.squeeze(0), scores

    def captions_variations(self, input_features, beam_sizes):
        """
        Function to generate the few captions for the corresponding encoded image
        using augmented beam search to provide the most optimal captions combination.

        Arguments:
            input_features (tensor): Extracted features from the encoder module.
                (batch_size, feature_pixels = 14x14, encoder_dim = 512)
            beam_sizes (list): List of beam_size values.

        Output:
            output (list): ordered list of captions from different beam sizes
        """
        output = []
        for beam_size in beam_sizes:
            img_features = torch.clone(input_features)
            img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
            k_prev_words = torch.zeros(beam_size, 1).long()
            sentences = k_prev_words
            top_k_scores = torch.zeros(beam_size, 1)
            completed_sentences = []
            completed_sentences_scores = []
            h, c = self.get_init_lstm_state(img_features)

            for step in range(config.MAX_STEPS):
                h, c, scores = self.get_scores(k_prev_words, img_features, h, c, top_k_scores)

                # artificially reduce the score to increase the diversity of captions between different beam_sizes
                if step < config.MIN_LENGTH:
                    random_row = random.randint(0, scores.size()[0] - 1)
                    scores[random_row][1] = -1e20

                # artificially reduce the score of words that have already been found in previous captions
                if len(output) > 0:
                    obtained_tokens = set(chain.from_iterable(output))
                    obtained_tokens.remove(0)
                    obtained_tokens.remove(1)
                    for token in obtained_tokens:
                        scores[0][token] = -1e10

                scores = F.log_softmax(scores, dim=1)

                if step == 0:
                    top_k_scores, top_k_words = scores[0].topk(beam_size, 0, True, True)
                else:
                    top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)

                prev_word_idxs = top_k_words / scores.size(1)
                next_word_idxs = top_k_words % scores.size(1)
                prev_word_idxs = prev_word_idxs.long()

                sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)

                incomplete_inds = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
                complete_inds = list(set(range(len(next_word_idxs))) - set(incomplete_inds))

                if len(complete_inds) > 0:
                    completed_sentences.extend(sentences[complete_inds].tolist())
                    completed_sentences_scores.extend(top_k_scores[complete_inds])

                beam_size -= len(complete_inds)
                if beam_size == 0:
                    break

                sentences = sentences[incomplete_inds]
                h = h[prev_word_idxs[incomplete_inds]]
                c = c[prev_word_idxs[incomplete_inds]]
                img_features = img_features[prev_word_idxs[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_idxs[incomplete_inds].unsqueeze(1)

            idx = completed_sentences_scores.index(max(completed_sentences_scores))
            sentence = completed_sentences[idx]
            output.append(sentence)
        return output
