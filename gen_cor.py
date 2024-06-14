import numpy as np
import torchvision
from imagecorruptions import corrupt
from imagecorruptions import get_corruption_names

dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
corruption_names = get_corruption_names()
mdict = {cname: [] for cname in corruption_names}
print(corruption_names)
images = []
labels = []

for image, label in dataset:
    image = np.array(image)
    label = np.array(label)
    images.append(image)
    labels.append(label)
    for corruption in corruption_names:
        for severity in range(5):
            corrupted = corrupt(image, corruption_name=corruption, severity=severity + 1)
            mdict[corruption].append(corrupted)

images = np.array(images)
labels = np.array(labels)
total_samples = len(dataset)

for corruption in mdict.keys():
    mdict[corruption] = np.array(mdict[corruption])
    mdict[corruption] = mdict[corruption].reshape(total_samples, 5, 32, 32, 3).transpose((1, 0, 2, 3, 4)).reshape(-1, 32, 32, 3)
    print(mdict[corruption].shape, mdict[corruption].dtype, labels.shape, labels.dtype)
    np.save(f'./data/{corruption}.npy', mdict[corruption])

np.save('./data/labels.npy', labels)
