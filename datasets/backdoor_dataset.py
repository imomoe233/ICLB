import sys
import bchlib
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import copy
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants

from encode_image import encode_image

tf.disable_v2_behavior()

class ReferenceImg(Dataset):

    def __init__(self, reference_file, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.target_input_array = np.load(reference_file)

        self.data = self.target_input_array['x']
        self.targets = self.target_input_array['y']

        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class BadEncoderDataset(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_file, indices, class_type, transform=None, bd_transform=None, ftt_transform=None):
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']

        self.trigger_input_array = np.load(trigger_file)
        self.target_input_array = np.load(reference_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']
        self.target_image_list = self.target_input_array['x']

        self.classes = class_type
        self.indices = indices
        self.transform = transform
        self.bd_transform = bd_transform
        self.ftt_transform = ftt_transform
        
        self.sess = tf.InteractiveSession(graph=tf.get_default_graph())
        model_path = 'ckpt/encoder_imagenet'
        self.model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], model_path)

    def __getitem__(self, index):
        img = self.data[self.indices[index]]
        img_copy = copy.deepcopy(img)
        backdoored_image = copy.deepcopy(img)
        
        img = Image.fromarray(img)
        '''original image'''
        if self.transform is not None:
            im_1 = self.transform(img)
        img_raw = self.bd_transform(img)
        '''generate backdoor image'''
        im_hidden, im_residual = encode_image(backdoored_image, self.model, self.sess)
        img_backdoor_list = []
        for i in range(len(self.target_image_list)):
            # getitem，提取的每一个shadow，都会添加上触发器，然后把shadow+触发器放到img_backdoor_list
            # 每个shadow对应的触发器都放在trigger_patch_list里了，所以可以直接把residual放到trigger_patch_list里
            # 对于每个样本 都计算residual并相加作为后门样本，修改⬇️
            # backdoored_image[:,:,:] = img_copy * self.trigger_mask_list[i] + self.trigger_patch_list[i][:]
            backdoored_image[:,:,:] = img_copy + im_residual
            img_backdoor =self.bd_transform(Image.fromarray(backdoored_image))
            img_backdoor_list.append(img_backdoor)

        target_image_list_return, target_img_1_list_return = [], []
        for i in range(len(self.target_image_list)):
            target_img = Image.fromarray(self.target_image_list[i])
            target_image = self.bd_transform(target_img)
            target_img_1 = self.ftt_transform(target_img)
            target_image_list_return.append(target_image)
            target_img_1_list_return.append(target_img_1)

        return img_raw, img_backdoor_list, target_image_list_return, target_img_1_list_return
    
    def __len__(self):
        return len(self.indices)
    
    def __del__(self):
        self.sess.close()


class BadEncoderTestBackdoor(Dataset):

    def __init__(self, numpy_file, trigger_file, reference_label, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y']


        self.trigger_input_array = np.load(trigger_file)

        self.trigger_patch_list = self.trigger_input_array['t']
        self.trigger_mask_list = self.trigger_input_array['tm']

        self.target_class = reference_label

        self.test_transform = transform
        
        self.sess = tf.InteractiveSession(graph=tf.get_default_graph())
        model_path = 'ckpt/encoder_imagenet'
        self.model = tf.saved_model.loader.load(self.sess, [tag_constants.SERVING], model_path)

    def __getitem__(self,index):
        img = copy.deepcopy(self.data[index])

        im_hidden, im_residual = encode_image(img, self.model, self.sess)
        # img[:] =img * self.trigger_mask_list[0] + self.trigger_patch_list[0][:]
        img[:] =img + im_residual
        img_backdoor =self.test_transform(Image.fromarray(img))
        return img_backdoor, self.target_class


    def __len__(self):
        return self.data.shape[0]

    def __del__(self):
        self.sess.close()

class CIFAR10CUSTOM(Dataset):

    def __init__(self, numpy_file, class_type, transform=None):
        """
        Args:
            numpy_file (string): Path to the numpy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.input_array = np.load(numpy_file)
        self.data = self.input_array['x']
        self.targets = self.input_array['y'][:,0].tolist()
        self.classes = class_type
        self.transform = transform
    def __len__(self):
        return self.data.shape[0]


class CIFAR10Pair(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            im_1 = self.transform(img)
            im_2 = self.transform(img)

        return im_1, im_2


class CIFAR10Mem(CIFAR10CUSTOM):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
