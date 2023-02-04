import json
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as td
from tqdm import tqdm

import config
import utils
from dataset import NamingDataset
from decoder import Decoder
from encoder import Wsl_encoder
from experiment import Experiment

log_file = open(config.LOG_FILE, 'w')
cudnn.benchmark = True
# torch.autograd.profiler.profile(enabled=False)
# torch.autograd.profiler.emit_nvtx(enabled=False)

train_set = NamingDataset(None, config.DATA_DIR)
val_set = NamingDataset(None, config.DATA_DIR, mode='val')

word_dict = json.load(open(os.path.join(config.DATA_DIR, config.WORD_DICT), 'r'))
vocabulary_size = len(word_dict)

# Creation of all Dependencies for Experiment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device = {device}")

print("Train caption verification: ", utils.generate_caption(train_set[0][1], word_dict))
print("Val caption verification: ", utils.generate_caption(val_set[0][1], word_dict))

encoder = Wsl_encoder().to(device)
decoder = Decoder(device, vocabulary_size, config.ENCODER_DIM, tf_ratio=config.TF_RATIO).to(device)
optimizer = optim.Adam(decoder.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, config.STEP_SIZE)
cross_entropy_loss = nn.CrossEntropyLoss().to(device)

# Create dataloaders for the training and validation set
train_loader = td.DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                             num_workers=config.NUM_WORKERS, pin_memory=True)
val_loader = td.DataLoader(val_set, batch_size=config.BATCH_SIZE * 2, shuffle=False,
                           num_workers=config.NUM_WORKERS, pin_memory=True)

# Creation of Experiment
exp = Experiment(config.START_EPOCH, encoder, decoder, optimizer, cross_entropy_loss, train_loader,
                 val_loader, word_dict, config.ALPHA_C, log_file, config.LOG_INTERVAL, device)

# Set variable to the location that the model is to be loaded from. Eg: 'models/model.pth.tar'
if config.MODEL_PATH:
    exp.load(config.MODEL_PATH)

print(f'Starting training from {exp.start_epoch} for {config.NUM_EPOCHS - exp.start_epoch + 1} epochs.')

for epoch in tqdm(range(exp.start_epoch, config.NUM_EPOCHS + 1)):
    model_file = os.path.join(config.DATA_DIR, f'model_{str(epoch)}.pth.tar')
    exp.train(epoch)  # Perform training on the complete dataset in batches
    exp.save(epoch, model_file)  # Save the current state of the model
    print('Saved model to ' + model_file)
    exp.validate(epoch)  # Perform validation after every epoch
    scheduler.step()

config.LOG_FILE.close()
