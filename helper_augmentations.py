import random
from random import randint
from random import shuffle
import torchvision

class SwapReferenceTest(object):
    def __call__(self, sample):
        prob = random.random()
        # Half chance to swap reference and test
        if prob > 0.5:
            trf_reference = sample['reference']
            trf_test = sample['test']
        else:
            trf_reference = sample['test']
            trf_test = sample['reference']
                
        return trf_reference, trf_test

class JitterGamma(object):
    def __call__(self, sample):
        prob = random.random()
        trf_reference = sample['reference']
        trf_test = sample['test']
        # Half chance to swap reference and test
        if prob > 0.5:
            gamma = random.random() + 0.1
            trf_reference = torchvision.transforms.functional.adjust_gamma(trf_reference, gamma)
            trf_test = torchvision.transforms.functional.adjust_gamma(trf_test, gamma)
                
        return trf_reference, trf_test