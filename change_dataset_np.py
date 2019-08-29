from torch.utils.data import Dataset
import numpy as np
import pickle

class ChangeDatasetNumpy(Dataset):
    """ChangeDataset Numpy Pickle Dataset"""

    def __init__(self, pickle_file, transform=None):
    
        # Load pickle file with Numpy dictionary
        f = open(pickle_file,"rb")
        self.data_dict= pickle.load(f)
        f.close()          
    
        self.transform = transform

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):        
        reference_PIL, test_PIL, label_PIL = self.data_dict[idx]
        sample = {'reference': reference_PIL, 'test': test_PIL, 'label': label_PIL}        

        # Handle Augmentations
        if self.transform:
            trf_reference = self.transform(sample['reference'])
            trf_test = self.transform(sample['test'])
            trf_label = self.transform(sample['label'])
            sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label}

        return sample