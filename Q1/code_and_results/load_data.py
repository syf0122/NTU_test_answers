from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

def load():
    # datasets
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )


    # get 0 and seven
    idx = (training_data.targets==7) | (training_data.targets==0)
    training_data.targets = training_data.targets[idx]
    training_data.data = training_data.data[idx]

    idx = (test_data.targets==7) | (test_data.targets==0)
    test_data.targets = test_data.targets[idx]
    test_data.data = test_data.data[idx]

    # Create DataLoaders for train and test data
    batch_size = 100
    train = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last = True)
    test = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last = True)
    return train, test

# train, test = load()
# # Plot to check if I used the right dataset
# fig = plt.figure()
# examples = enumerate(train)
# batch_idx, (example_data, example_targets) = next(examples)
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# plt.show()
