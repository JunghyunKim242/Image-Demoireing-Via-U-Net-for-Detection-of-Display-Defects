import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import  torchvision.transforms.functional as TF
import torchvision
import cv2
## 474747474747474747


def loader_16bit(path ):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.mean(img, axis=-1)
    return img


def loader_8bit(path):
    # img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    # img =cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # img = np.stack((img,) * 3, axis=-1)
    return img


def loader_8bit_color(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def loader_8bit_gray2rgb(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = np.stack((img) * 3, axis=-1)
    return img


class Captured_Moire_dataset_train(Dataset):
    def __init__(self, root,  loader = loader_8bit ):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.patchsize = 512
        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

        # image_names2 = [".".join(i.split(".")[:-1]) for i in image_names2]
        # self.labels = image_names1 + image_names2


    def __getitem__(self, index):
        # ##############################################
        # #원래꺼
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        # 16bit
        # moire_img_path = self.moire_images[index]
        # moire = self.loader_16bit(moire_img_path)
        # moire = self.transforms(moire)
        # moire = moire / 4095
        # moire = moire.float()

        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)

        # 16bit
        # clear_img_path = self.clear_images[index]
        # clear = self.loader_16bit(clear_img_path)
        # clear = self.transforms(clear)
        # clear = clear / 65535
        # clear = clear.float()

        moire = TF.crop(moire, 170, 40, 2000, 1000)
        clear = TF.crop(clear, 170, 40, 2000, 1000) # 2000,1000 size

        i, j, h, w = transforms.RandomCrop.get_params( moire, output_size=(self.patchsize,self.patchsize) )
        moire = TF.crop(moire, i, j, h, w)
        clear = TF.crop(clear, i, j, h, w)


        ##############################################
        # range를 키웠을 때
        # clear_img_path = self.clear_images[index]
        # clear = self.loader(clear_img_path)
        # clear = clear.astype(np.float32)
        # clear = clear[170:170+2000,40:40+1000]
        # # clear = ((clear-130) * (54/19)) + 130
        # clear = ((clear-130) * (2.2)) + 125
        # clear = clear / 255
        # clear = self.transforms(clear)
        # clear = TF.crop(clear, i, j, h, w)
        ###############################################


        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


class Captured_Moire_dataset_test(Dataset):
    def __init__(self, root, loader = loader_8bit):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.centercrop = transforms.CenterCrop(512)

        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

        # image_names2 = [".".join(i.split(".")[:-1]) for i in image_names2]
        # self.labels = image_names1 + image_names2

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        ## 16 비트
        # moire_img_path = self.moire_images[index]
        # moire = self.loader_16bit(moire_img_path)
        # moire = self.transforms(moire)
        # moire = moire / 4095
        # moire = moire.float()

        #claer
        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)

        # 16bit
        # clear_img_path = self.clear_images[index]
        # clear = self.loader_16bit(clear_img_path)
        # clear = self.transforms(clear)
        # clear = clear / 65535
        # clear = clear.float()

        moire = TF.crop(moire, 170, 40, 2000, 1000)
        clear = TF.crop(clear, 170, 40, 2000, 1000) # 2000,1000 size

        ################
        # range를 키웠을때
        # clear_img_path = self.clear_images[index]
        # clear = self.loader(clear_img_path)
        # clear = clear.astype(np.float32)
        # clear = clear[170:170+2000,40:40+1000]
        # clear = ((clear-130) * (2.2)) + 125
        # clear = clear / 255
        # clear = self.transforms(clear)
        ################

        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)












#########################
class Captured_Moire_dataset_train_MHA(Dataset):
    def __init__(self, root,  loader = loader_8bit ):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.patchsize = 256
        self.centercrop = transforms.CenterCrop(512)

        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1
        self.lens = len(self.moire_images)

    def __getitem__(self, index):
        # #원래꺼
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)


        ## 16 bit
        # moire_img_path = self.moire_images[index]
        # moire = self.loader_16bit(moire_img_path)
        # moire = self.transforms(moire)
        # moire = moire / 4095
        # moire = moire.float()

        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)
        ##16 bit
        # clear_img_path = self.clear_images[index]
        # clear = self.loader_16bit(clear_img_path)
        # clear = self.transforms(clear)
        # clear = (clear /65535)*4095
        # clear = np.floor(clear)
        # clear = clear/4095
        # clear = clear.float()

        moire = TF.crop(moire, 170, 28, 2000, 1024)
        clear = TF.crop(clear, 170, 28, 2000, 1024)

        i, j, h, w = transforms.RandomCrop.get_params( moire, output_size=(self.patchsize,self.patchsize) )
        moire = TF.crop(moire, i, j, h, w)
        clear = TF.crop(clear, i, j, h, w)

        ## center crop
        # moire = self.centercrop(moire)
        # clear = self.centercrop(clear)

        ### image_augmentation
        randn = torch.rand(1)
        alpha = ( (randn*2-1) * 0.2) * (1/255)
        moire = moire + alpha


        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        # return len(self.moire_images)
        return self.lens


class Captured_Moire_dataset_test_MHA(Dataset):
    def __init__(self, root, loader = loader_8bit):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.centercrop = transforms.CenterCrop(512)

        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1
        self.lens = len(self.moire_images)

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)
        ## 16 bit
        # moire_img_path = self.moire_images[index]
        # moire = self.loader_16bit(moire_img_path)
        # moire = self.transforms(moire)
        # moire = moire / 4095
        # moire = moire.float()


        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)
        ## 16 bit
        # clear_img_path = self.clear_images[index]
        # clear = self.loader_16bit(clear_img_path)
        # clear = self.transforms(clear)
        # clear = (clear /65535)*4095
        # clear = np.floor(clear)
        # clear = clear/4095
        # clear = clear.float()


        moire = TF.crop(moire, 170, 28, 2000, 1024)
        clear = TF.crop(clear, 170, 28, 2000, 1024)

        # moire = self.centercrop(moire)
        # clear = self.centercrop(clear)

        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return self.lens
#########################











##16bit
class Captured_Moire_dataset_train_16bit(Dataset):
    def __init__(self, root,  loader = loader_8bit ):
        moire_data_root = os.path.join(root, 'moire_16bit')
        clear_data_root = os.path.join(root, 'clean_16bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.patchsize = 512
        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

    def __getitem__(self, index):
        # ##############################################
        # 16bit
        moire_img_path = self.moire_images[index]
        moire = self.loader_16bit(moire_img_path)
        moire = self.transforms(moire)
        moire = moire / 4095
        moire = moire.float()

        # 16bit
        # clear_img_path = self.clear_images[index]
        # clear = self.loader_16bit(clear_img_path)
        # clear = self.transforms(clear)
        # clear = clear / 65535
        # clear = clear.float()

        moire = TF.crop(moire, 170, 40, 2000, 1000)
        # clear = TF.crop(clear, 170, 40, 2000, 1000) # 2000,1000 size

        i, j, h, w = transforms.RandomCrop.get_params( moire, output_size=(self.patchsize,self.patchsize) )
        moire = TF.crop(moire, i, j, h, w)
        # clear = TF.crop(clear, i, j, h, w)

        ##############################################
        # range를 키웠을 때
        clear_img_path = self.clear_images[index]
        clear = self.loader_16bit(clear_img_path)
        clear = clear.astype(np.float32)
        clear = clear[170:170+2000,40:40+1000]
        # clear = ((clear-130) * (54/19)) + 130
        # clear = ((clear-130) * (2.2)) + 125
        clear = (clear/65535)*4095
        clear = np.floor(clear)
        clear = ((clear-2087) * 2.2) + 2000
        clear = clear / 4095
        clear = self.transforms(clear)
        clear = TF.crop(clear, i, j, h, w)
        ###############################################


        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


class Captured_Moire_dataset_test_16bit(Dataset):
    def __init__(self, root, loader = loader_8bit):
        moire_data_root = os.path.join(root, 'moire_16bit')
        clear_data_root = os.path.join(root, 'clean_16bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.centercrop = transforms.CenterCrop(512)

        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1
        # image_names2 = [".".join(i.split(".")[:-1]) for i in image_names2]
        # self.labels = image_names1 + image_names2

    def __getitem__(self, index):

        ## 16 비트
        moire_img_path = self.moire_images[index]
        moire = self.loader_16bit(moire_img_path)
        moire = self.transforms(moire)
        moire = moire / 4095
        moire = moire.float()

        moire = TF.crop(moire, 170, 40, 2000, 1000)
        # clear = TF.crop(clear, 170, 40, 2000, 1000) # 2000,1000 size

        ##############################################
        # range를 키웠을 때
        clear_img_path = self.clear_images[index]
        clear = self.loader_16bit(clear_img_path)
        clear = clear.astype(np.float32)
        clear = clear[170:170+2000,40:40+1000]
        # clear = ((clear-130) * (54/19)) + 130
        # clear = ((clear-130) * (2.2)) + 125
        clear = (clear/65535)*4095
        clear = np.floor(clear)
        clear = ((clear-2087) * 2.2) + 2000
        clear = clear / 4095
        clear = self.transforms(clear)
        ###############################################

        label = self.labels[index]
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)













#########################
#########################
#mbcnn
class Captured_Moire_dataset_train_MBCNN(Dataset):
    def __init__(self, root,  loader = loader_8bit_gray2rgb ):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.patchsize = 256
        self.transforms2 = transforms.Resize((self.patchsize//2,self.patchsize//2))
        self.transforms3 = transforms.Resize((self.patchsize//4, self.patchsize//4))
        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1


    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)

        i, j, h, w = transforms.RandomCrop.get_params( moire, output_size=(self.patchsize,self.patchsize) )
        moire = TF.crop(moire, i, j, h, w)
        clear = TF.crop(clear, i, j, h, w)

        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear)

        clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)


#####
class Captured_Moire_dataset_test_MBCNN(Dataset):
    def __init__(self, root, loader = loader_8bit_gray2rgb):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.patchsize = 512
        self.transforms = transforms.ToTensor()
        self.transforms2 = transforms.Resize((self.patchsize//2,self.patchsize//2))
        self.transforms3 = transforms.Resize((self.patchsize//4, self.patchsize//4))

        self.centercrop = transforms.CenterCrop(512)

        self.loader = loader
        self.loader_16bit = loader_16bit

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)

        moire = self.centercrop(moire)
        clear = self.centercrop(clear)

        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear)

        clear_list = [clear3, clear2, clear]

        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)

#mbcnn


######mbcnn aim 2019 dataset
#mbcnn
class AIM19_train_MBCNN(Dataset):
    def __init__(self, root,  loader = loader_8bit_color ):
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clean')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.patchsize = 128
        self.transforms2 = transforms.Resize((self.patchsize//2,self.patchsize//2))
        self.transforms3 = transforms.Resize((self.patchsize//4, self.patchsize//4))
        self.loader = loader

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1


    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)

        i, j, h, w = transforms.RandomCrop.get_params( moire, output_size=(self.patchsize,self.patchsize) )
        moire = TF.crop(moire, i, j, h, w)
        clear = TF.crop(clear, i, j, h, w)

        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear)

        clear_list = [clear3, clear2, clear]
        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)


class AIM19_test_MBCNN(Dataset):
    def __init__(self, root, loader = loader_8bit_color):
        moire_data_root = os.path.join(root, 'moire')
        clear_data_root = os.path.join(root, 'clean')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.patchsize   = 1024
        self.transforms  = transforms.ToTensor()
        self.transforms2 = transforms.Resize((self.patchsize//2,self.patchsize//2))
        self.transforms3 = transforms.Resize((self.patchsize//4, self.patchsize//4))

        self.loader = loader
        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)
        clear2 = self.transforms2(clear)
        clear3 = self.transforms3(clear)

        clear_list = [clear3, clear2, clear]

        label = self.labels[index]
        return moire, clear_list, label

    def __len__(self):
        return len(self.moire_images)











######################
######################
##test mode
class Captured_Moire_dataset_testmode(Dataset):
    def __init__(self, root, loader = loader_8bit):
        moire_data_root = os.path.join(root, 'moire_12bit')
        clear_data_root = os.path.join(root, 'clean_12bit')

        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))

        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]

        self.transforms = transforms.ToTensor()
        self.centercrop = transforms.CenterCrop(512)

        self.loader = loader
        self.loader_16bit = loader_16bit

        self.centercrop = transforms.CenterCrop(512)

        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

    def __getitem__(self, index):
        # moire_img_path = self.moire_images[index]
        # moire = self.loader(moire_img_path)
        # moire = self.transforms(moire)

        # moire_img_path = self.moire_images[index]
        # moire = self.loader_16bit(moire_img_path)
        # moire = self.transforms(moire)
        # moire = moire / 4095
        # moire = moire.float()

        #12 to 8bit
        moire_img_path = self.moire_images[index]
        moire = self.loader_16bit(moire_img_path)
        moire = self.transforms(moire)
        moire = (moire /4095)*255
        moire = np.floor(moire)
        moire = moire/255
        moire = moire.float()

        #원래 clear
        # clear_img_path = self.clear_images[index]
        # clear = self.loader(clear_img_path)
        # clear = self.transforms(clear)

        # clear_img_path = self.clear_images[index]
        # clear = self.loader_16bit(clear_img_path)
        # clear = self.transforms(clear)
        # clear = (clear /65535)*4095
        # clear = np.floor(clear)
        # clear = clear/4095
        # clear = clear.float()

        # 12 to 8bit
        clear_img_path = self.clear_images[index]
        clear = self.loader_16bit(clear_img_path)
        clear = self.transforms(clear)
        clear = (clear /65535)*255
        clear = np.floor(clear)
        clear = clear/255
        clear = clear.float()


        moire = TF.crop(moire, 170, 28, 2000, 1024)
        clear = TF.crop(clear, 170, 28, 2000, 1024)

        # moire = self.centercrop(moire)
        # clear = self.centercrop(clear)


        label = self.labels[index]

        # return moire, clear, label
        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)













###
class Captured_Moire_dataset_testmode_MBCNN(Dataset):
    def __init__(self, root, loader = loader_8bit_gray2rgb):
        moire_data_root = os.path.join(root, 'moire_8bit')
        clear_data_root = os.path.join(root, 'clean_8bit')
        image_names1 = sorted(os.listdir(moire_data_root))
        image_names2 = sorted(os.listdir(clear_data_root))
        self.moire_images = [os.path.join(moire_data_root, x ) for x in image_names1]
        self.clear_images = [os.path.join(clear_data_root, x ) for x in image_names2]
        self.transforms = transforms.ToTensor()
        self.loader = loader
        self.loader_16bit = loader_16bit
        self.centercrop = transforms.CenterCrop(512)
        image_names1 = [".".join(i.split(".")[:-1]) for i in image_names1]
        self.labels = image_names1

    def __getitem__(self, index):
        moire_img_path = self.moire_images[index]
        moire = self.loader(moire_img_path)
        moire = self.transforms(moire)

        #원래 clear
        clear_img_path = self.clear_images[index]
        clear = self.loader(clear_img_path)
        clear = self.transforms(clear)

        moire = TF.crop(moire, 170, 40, 2000, 1000)
        clear = TF.crop(clear, 170, 40, 2000, 1000)

        label = self.labels[index]

        return moire, clear, label

    def __len__(self):
        return len(self.moire_images)


# range를 키웠을 때
# clear_img_path = self.clear_images[index]
# clear = self.loader_16bit(clear_img_path)
# clear = clear.astype(np.float32)
# clear = clear[170:170+2000,40:40+1000]
# # clear = ((clear-130) * (54/19)) + 130
# # clear = ((clear-130) * (2.2)) + 125
# clear = (clear/65535)*4095
# clear = np.floor(clear)
# clear = ((clear-2087) * 2.2) + 2000
# clear = clear / 4095
# clear = self.transforms(clear)