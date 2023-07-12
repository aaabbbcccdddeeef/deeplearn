from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms


class AtomicDataset(Dataset):
    def __init__(self,root,image_size):
        Dataset.__init__(self)
        self.dataset=datasets.ImageFolder(root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    def __getitem__(self,index):
        return self.dataset[index]
    def __len__(self):
        return len(self.dataset)

    def toBatchLoader(self,batch_size):
        return DataLoader(self,batch_size=batch_size, shuffle=False)

