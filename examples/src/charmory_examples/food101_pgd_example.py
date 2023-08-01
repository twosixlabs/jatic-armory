"""
This file serves as an example of ART-less projected gradient descent attacks. This file provides the general structure for a training and evaluation pipeline on
the Food101 Dataset
"""


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = "/home/rahul/cache"
BATCH_SIZE = 16
STEP_VALUE_TRAIN = 5
STEP_VALUE_TEST = 5
NUM_EPOCHS = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()


def projected_gradient_descent(
    model,
    x,
    y,
    epsilon,
    alpha,
    num_steps,
    random_start,
):
    """
    Adapted from torchattacks PGD code (https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/pgd.html#PGD)
    """
    images = x.clone().detach().to(DEVICE)
    labels = y.clone().detach().to(DEVICE)
    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -epsilon, epsilon
        )
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(num_steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        cost = loss(outputs, labels)

        grad = torch.autograd.grad(
            cost, adv_images, retain_graph=False, create_graph=False
        )[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


class Net(nn.Module):
    """
    This is a simple CNN for food101 and does not achieve SotA performance
    """

    def __init__(self):
        # Model architecture loosely adapted from the cifar baseline_model
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 32 * 256, 256)
        self.fc2 = nn.Linear(256, 101)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


model = Net()
model.to(DEVICE)
train = datasets.Food101(
    ROOT,
    split="train",
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(512, 512)),
        ]
    ),
    download=False,
)
train_mask = list(range(0, 75750, STEP_VALUE_TRAIN))
masked_training_set = torch.utils.data.Subset(train, train_mask)
train_loader = DataLoader(
    masked_training_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
)
model.train()
print("Training Model...")
optimiser = optim.SGD(model.parameters(), lr=0.003)
loss_fn = nn.CrossEntropyLoss()

# Training Loop
for epoch in range(NUM_EPOCHS):
    print("Epoch: " + str(epoch + 1))
    i = 0
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        i += 1
        optimiser.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimiser.step()
        print(str(i) + " Batches Completed")


# Testing Loop
test = datasets.Food101(
    ROOT,
    split="test",
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=(512, 512)),
        ]
    ),
    download=False,
)

test_mask = list(range(0, 25250, STEP_VALUE_TEST))
masked_testing_set = torch.utils.data.Subset(test, test_mask)
test_loader = DataLoader(masked_testing_set, batch_size=BATCH_SIZE, num_workers=4)

correct_pred_adv = 0

model.eval()
print("Performing Adversarial Testing...")
i = 0
for x, y in test_loader:
    i += 1
    x = x.to(DEVICE)
    y = y.to(DEVICE)
    x_adversarial = projected_gradient_descent(
        model,
        x,
        y,
        num_steps=20,
        random_start=False,
        epsilon=0.031,
        alpha=0.007,
    )
    outputs_adv = model(x_adversarial)
    _, predictions_adv = torch.max(outputs_adv, 1)
    correct_pred = torch.sum(predictions_adv == y)
    correct_pred_adv += correct_pred.item()
    print(str(i) + " Batches Completed")

adv_accuracy = (100 * correct_pred_adv) / len(test_mask)
print("Adversarial Test Accuracy: " + str(adv_accuracy) + "%")
