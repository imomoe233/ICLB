from PIL import Image
import numpy as np
import pickle


dataset_path = "X:\Directory\code\dataset\cifar-100-python"

def get_benign_cifar10():
    def load_batch(file):
        with open(f'{file}', 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            data = batch['data'].reshape(10000, 3, 32, 32)
            # 将数据重新排列为 [10000, 32, 32, 3] (即将通道放在最后)
            data = data.transpose(0, 2, 3, 1)
        return data

    # 加载所有数据批次
    #batches = [load_batch(f'{dataset_path}/test_batch')]
    batches = [load_batch(f'{dataset_path}/data_batch_{i}') for i in range(1, 6)]
    # 堆叠所有数据
    benign_cifar_train_data = np.vstack(batches)

    # batches = [load_batch(f'{dataset_path}/test_batch')]

    # 转换为PIL图像列表
    benign = np.array([np.array(Image.fromarray(image)) for image in benign_cifar_train_data]).astype(np.uint8)
    #images_resized = np.array([np.array(Image.fromarray(image).resize((224, 224), Image.ANTIALIAS)) for image in benign])
    #print(images_resized.shape)
    #Image.fromarray(images_resized[0]).show()
    
    return benign

def get_benign_cifar10_train_label():

    def load_labels(file):
        with open(f'{file}', 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        return batch['labels']

    # 加载所有标签批次
    #labels = [load_labels(f'{dataset_path}/test_batch')]
    labels = [load_labels(f'{dataset_path}/data_batch_{i}') for i in range(1, 6)]
    # 合并所有标签
    benign_cifar10_train_label = sum(labels, [])

    return benign_cifar10_train_label


data = get_benign_cifar10()
print(type(data))
print(data.shape)
# print(data)

label = get_benign_cifar10_train_label()
label = np.array([[i] for i in label]).astype(np.uint8)

print(label)


np.savez(r'D:\code\dataset/cifar10-SAB/train.npz', x = data, y = label)