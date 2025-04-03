

def load_mnist():
    dset = torchvision.datasets.MNIST(root=f"{raw_data_path}/mnist", download=True, train=False, transform=torchvision.transforms.ToTensor())
    labels = []
    for i in range(len(dset)):
        img, label = dset[i]
        labels.append(label)
    labels = np.array(labels)
    return dset, labels


