from torch.utils.data import Dataset
import numpy as np
import pickle
import torchvision

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
            # Handle label class differently
            trf_label = sample['label']
            # Dont do Normalize on label, all the other transformations apply...
            for t in self.transform.transforms:
                if not isinstance(t, torchvision.transforms.transforms.Normalize):
                    # ToTensor divide every result by 255
                    # https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#to_tensor
                    if isinstance(t, torchvision.transforms.transforms.ToTensor):
                        trf_label = t(trf_label) * 255.0
                    else:
                        trf_label = t(trf_label)                
            sample = {'reference': trf_reference, 'test': trf_test, 'label': trf_label}

        return sample