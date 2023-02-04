
from collections import Counter
import json

import numpy as np
import config
from PIL import Image
import matplotlib.pyplot as plt
import torchvision as tv
import torch
from math import ceil
import skimage
import skimage.transform
import matplotlib.cm as cm


# Transform to be applied on the image prior to processing - Used for the data visualization step
data_transforms = tv.transforms.Compose([
    tv.transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])


class AverageMeter(object):
    '''
    Taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    This class can be used to maintain average statistics over multiple iterations of training and validation.
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    '''
    Function to compute the accuracy of prediction.

    Arguments:
        preds: Predictions of the network
        targets: Expected output of the network
        k (int): Top k accuracy to be determined
                 Example: k=1 -> Top 1 accuracy
                          k=5 -> Top 5 accuracy
    '''
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def calculate_caption_lengths(word_dict, captions):
    '''
    Calculate the length of the caption excluding the start, end and padding tokens in the caption.

    Arguments:
        word_dict (dict): Dictionary of words (vocabulary)
        captions (list): List of encoded captions where each entry in the encoded caption
            is corresponding index from word_dict
    '''
    lengths = 0
    for caption_tokens in captions:
        for token in caption_tokens:
            if token in (word_dict['<start>'], word_dict['<eos>'], word_dict['<pad>']):
                continue
            else:
                lengths += 1
    return lengths


def generate_caption(enc_caption, word_dict):
    '''
    Function to create the caption sentence from the encoded caption using the word dictionary

    Arguments:
        enc_caption (list): Encoded caption in terms of dictionary indices
        word_dict (dict): Dictionary of words (vocabulary)
    '''

    # Using the dictionary, convert the encoded caption to normal words
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    enc_caption = enc_caption.to('cpu').tolist()
    for word_idx in enc_caption:
        if word_idx == word_dict['<start>']:
            continue
        if word_idx == word_dict['<eos>']:
            break
        sentence_tokens.append(token_dict[word_idx])

    # Creation of a sentence from the list of words
    caption = ''
    for word in sentence_tokens:
        if word is sentence_tokens[len(sentence_tokens) - 1]:
            caption = caption + word + '.'
        else:
            caption = caption + word + ' '

    return caption.capitalize()


def process_caption_tokens(caption_tokens, word_dict, max_length):
    '''
    Function to encode the list of words in the caption into a corresponding list of their keys in the word dictionary.

    Arguments:
        caption_tokens (list): List of words to be processed into dictionary keys
        word_dict (dict): Dictionary of words to be used to encode list of tokens
        max_length (int): Maximum caption length

    Example:
        caption_tokens = ['a', 'woman', 'standing']
        word_dict = {0:'<start>', 1:'<eos>', 3:'<pad>', 6:'a', 10:'woman', 30:'standing'}
        max_length = 5
        OUT = [0, 6, 10, 30, 3, 3, 1]
    '''
    captions = []
    for tokens in caption_tokens:
        token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>'] for token in tokens]
        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']] +
            [word_dict['<pad>']] * (max_length - len(tokens)))

    return captions


def pil_loader(path):
    '''
    Load an image from the specified path and convert to RGB

    Arguments:
        path (str): Complete path of the image that is to be loaded
    '''
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def myimshow(image, ax=plt):
    '''
    Funtion to display an input image

    Arguments:
        image : Tensor or array of image pixel values
    '''

    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h


def clean_mask(mask, k):
    '''
    Function to masks the duplicated elements that constitute a duplicated group.

    Arguments:
        mask: (numpy.array) possible duplicated group
        k (int): length of possible duplicated group
    '''
    delta = np.diff(np.r_[False, mask, np.ones(k - 1, bool), False].view(np.int8))
    edges = np.flatnonzero(delta).reshape(-1, 2)
    lengths = edges[:, 1] - edges[:, 0]
    delta[edges[lengths < k, :]] = 0
    return delta[:-k].cumsum(dtype=np.int8).view(bool)


def remove_words_duplicates(data, kmax):
    '''
    Function to remove consecutively repeated groups of words in caption.

    Arguments:
        data: (list) Instance of the trained Encoder for encoding of images
        kmax (int): max possible length of possible duplicated group
    '''
    data = np.asarray(data)
    kmax = min(kmax, data.size // 2)
    remove = np.zeros(data.shape, dtype=np.bool)
    for k in range(kmax, 0, -1):
        remove[k:] |= clean_mask(data[k:] == data[:-k], k)
    return data[~remove]


def generate_image_captions(encoder, decoder, img_path, word_dict, axes, beam_sizes=[3]):
    '''
    Function to display the image along with the resultant predicted captions.

    Arguments:
        encoder: Instance of the trained Encoder for encoding of images
        decoder: Instance of the trained Decoder for caption prediction from encoded image
        img_path (str): Complete path of the image to be visualized
        word_dict (dict): Dictionary of words (vocabulary)
        axes: axes for plotting
        beam_sizes (list): List of beam_sizes values for beam search. Default = [3]
    '''
    # Load the image and transform it
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    # Get the caption from the trained network
    img_features = encoder(img)

    # If a single caption is needed, get it for a given value beam_size
    if len(beam_sizes) == 1:
        img_features = img_features.expand(beam_sizes[0], img_features.size(1), img_features.size(2))
        output = [decoder.caption(img_features, beam_sizes[0])[0]]
    else:
        output = decoder.captions_variations(img_features, config.BEAM_SIZES)

    # Using the dictionary, convert the encoded caption to normal words
    token_dict = {idx: word for word, idx in word_dict.items()}
    for i, sentence in enumerate(output):
        sentence = remove_words_duplicates(sentence, config.KMAX)
        sentence_tokens = []
        for word_idx in sentence:
            sentence_tokens.append(token_dict[word_idx])
        # Creation of a sentence from the list of words
        caption = ''
        for word in sentence_tokens[1:-1]:
            caption = caption + word + ' '
        caption = caption[:-1].capitalize() + '.'

        # Resizing image for a standard display
        img = Image.open(img_path)
        w, h = img.size
        if w > h:
            w = w * 256 / h
            h = 256
        else:
            h = h * 256 / w
            w = 256
        left = (w - 224) / 2
        top = (h - 224) / 2
        resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
        img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
        img = img.astype('float32') / 255

        if len(beam_sizes) == 1:
            axes.imshow(img)
            axes.set_title(caption)
            axes.axis('off')
        else:
            axes[i].imshow(img)
            axes[i].set_title(caption)
            axes[i].axis('off')


def generate_caption_visualization(encoder, decoder, img_path, word_dict, beam_size=3):
    '''
    Function to visualize the step by step development of the caption along
    with the corresponding attention component visualization.

    Arguments:
        encoder: Instance of the trained Encoder for encoding of images
        decoder: Instance of the trained Decoder for caption prediction from encoded image
        img_path (str): Complete path of the image to be visualized
        word_dict (dict): Dictionary of words (vocabulary)
        beam_size (int): Number of top candidates to consider for beam search. Default = 3
    '''

    # Load the image and transform it
    img = pil_loader(img_path)
    img = data_transforms(img)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)

    # Get the caption and the corresponding attention weights from the trained network
    img_features = encoder(img)
    img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
    sentence, alpha = decoder.caption(img_features, beam_size)

    # Using the dictionary, convert the encoded caption to normal words
    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break

    # Resizing image for a standard display
    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)

    num_words = len(sentence_tokens)
    w = np.round(np.sqrt(num_words))
    h = np.ceil(np.float32(num_words) / w)
    alpha = torch.tensor(alpha)

    # Plot the different attention weighted versions of the original image along
    # with the resultant caption word prediction
    f = plt.figure(figsize=(8, 9))
    plot_height = ceil((num_words + 3) / 4.0)
    ax1 = f.add_subplot(4, plot_height, 1)
    plt.imshow(img)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = f.add_subplot(4, plot_height, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img)

        # if encoder.network == 'vgg19':
        #     shape_size = 14
        # else:
        shape_size = 10

        alpha_img = skimage.transform.pyramid_expand(alpha[idx, :].reshape(shape_size, shape_size),
                                                     upscale=32, sigma=20)

        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


def generate_json_data(split_path, data_path, max_captions_per_image):
    split = json.load(open(split_path, 'r'))
    word_count = Counter()

    train_img_paths = []
    train_caption_tokens = []
    validation_img_paths = []
    validation_caption_tokens = []

    max_length = 0
    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            if img['split'] == 'train':
                train_img_paths.append(data_path + '/imgs/' + img['filepath'] + '/' + img['filename'])
                train_caption_tokens.append(sentence['tokens'])
            elif img['split'] == 'val':
                validation_img_paths.append(data_path + '/imgs/' + img['filepath'] + '/' + img['filename'])
                validation_caption_tokens.append(sentence['tokens'])
            max_length = max(max_length, len(sentence['tokens']))
            word_count.update(sentence['tokens'])

    words = [word for word in word_count.keys() if word_count[word] >= 5]
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}
    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3

    with open(data_path + '/word_dict.json', 'w') as f:
        json.dump(word_dict, f)

    train_captions = process_caption_tokens(train_caption_tokens, word_dict, max_length)
    validation_captions = process_caption_tokens(validation_caption_tokens, word_dict, max_length)

    with open(data_path + '/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path + '/val_img_paths.json', 'w') as f:
        json.dump(validation_img_paths, f)
    with open(data_path + '/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions.json', 'w') as f:
        json.dump(validation_captions, f)
