
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
# import numpy 
import os
import glob
trainpath = ".\\Datasets\\train\\"
testpath = ".\\Datasets\\test\\"
classes = os.listdir(trainpath)
train_imgs = []
test_imgs  = []

for _class in classes:
    train_imgs += glob.glob(trainpath + _class + '\*.png')
    test_imgs += glob.glob(testpath + _class + '\*.png')


cifar_transforms_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

cifar_transforms_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
class CIFAR10Dataset(Dataset):
    
    def __init__(self, imgs_list, classes, transforms=None):
        super(CIFAR10Dataset, self).__init__()
        self.imgs_list = imgs_list
        self.class_to_int = {classes[i] : i for i in range(len(classes))}
        self.transforms = transforms
    def __getitem__(self, index):
    
        image_path = self.imgs_list[index]
        
        # Reading image
        image = Image.open(image_path)
        
        # Retriving class label
        label = image_path.split("\\")[-2]
        label = self.class_to_int[label]
        # print(self.class_to_int)
        # Applying transforms on image
        if self.transforms is not None:
            image = self.transforms(image)
        else:
            image = transforms.ToTensor()(image)
        return image, label
        

    def __len__(self):
        return len(self.imgs_list)
    
train_dataset = CIFAR10Dataset(imgs_list = train_imgs, classes = classes, transforms = cifar_transforms_train)
test_dataset = CIFAR10Dataset(imgs_list = test_imgs, classes = classes, transforms = cifar_transforms_val)
