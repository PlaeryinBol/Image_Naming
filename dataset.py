import json
import torch
import torch.utils.data as td
import torchvision as tv
from multiprocessing import Manager
from PIL import Image
import config


class NamingDataset(td.Dataset):
    def __init__(self, transform, data_path,
                 mode='train', image_size=(config.IMG_SIZE, config.IMG_SIZE)):
        super(NamingDataset, self).__init__()
        self.manager = Manager()
        self.mode = mode
        if transform is not None:
            self.transform = transform
        else:
            self.transform = tv.transforms.Compose([
                             tv.transforms.ToTensor(),
                             tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ])
        self.image_size = image_size
        if self.mode == 'train':
            self.train_data = self.manager.dict(json.load(open(data_path + f'/paths_captions_{mode}.json', 'r')))
        else:
            self.val_data = self.manager.dict(json.load(open(data_path + f'/paths_captions_{mode}.json', 'r')))

    def __getitem__(self, index):
        # Access the image at the corresponding location and transform it as specified
        # For training, return the image and its corresponding captions
        index = str(index)
        if self.mode == 'train':
            img_path = self.train_data[index]['path']
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            img = self.transform(img)
            return img, torch.tensor(self.train_data[index]['caption'])
        else:
            img_path = self.val_data[index]['path']
            with open(img_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
            img = self.transform(img)
            # For validation, return the image and all the captions associated with that image in the complete dataset
            matching_idxs = [idx for idx, p in enumerate(self.val_data) if self.val_data[p]['path'] == img_path]
            all_captions = [self.val_data[str(idx)]['caption'] for idx in matching_idxs]
            return torch.FloatTensor(img), torch.tensor(self.val_data[index]['caption']), torch.tensor(all_captions)

    def __len__(self):
        if self.mode == 'train':
            return len(self.data)
        else:
            return len(self.val_data)
