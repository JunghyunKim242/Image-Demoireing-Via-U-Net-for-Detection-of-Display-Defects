import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from Util.util_collections import Time2Str, PSNR_tensor,PSNR_tensor_255, PSNR_tensor_4095,tensor2img2imwrite_4095
from data_loader.dataset import Captured_Moire_dataset_testmode_MBCNN
from torchnet import meter
# from skimage.metrics import peak_signal_noise_ratio
import torchvision.utils as vutils
import time
import math


def test_MBCNN(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    version = "_"+"Test"
    NET = "_"+"MBCNN(64)"
    dataset = "_"+"moireweak"
    bit = "_"+"8bit"
    trainsize_testsize = "_"+"train:256_test:512"
    detail = "_"+ "GT_resizing_None"
    name = version + NET + dataset + bit + trainsize_testsize + detail + "_"
    args.save_prefix = "/databse4/jhkim/PTHfolderIEEEACCESS/"

    args.save_prefix = args.save_prefix + Time2Str() + name
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)

    print('torch devices = ', args.device)
    print('save_path = ', args.save_prefix)

    Moiredata_test = Captured_Moire_dataset_testmode_MBCNN(args.testmode_path)
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=args.batchsize_testmode,
                                 num_workers=args.num_worker)

    model = nn.DataParallel(model)
    model = model.to(args.device)

    args.Test_pretrained_path = '/databse4/jhkim/PTHfolderIEEEACCESS/00_pthfolder/_Train_MBCNN(64)_week_moire_450_8bit_train:256_test:512_siglescale_psnr_54.9403.pth'
    if args.Test_pretrained_path:
        file_extnsion = (args.Test_pretrained_path.split(".")[-1])
        if file_extnsion =='pth': # statedict
            print('pretrained = \t',args.Test_pretrained_path)
            model.load_state_dict(torch.load(args.Test_pretrained_path),strict = False)
        elif file_extnsion =='tar':
            print('pretrained = \t',args.Test_pretrained_path)
            checkpoint = torch.load(args.Test_pretrained_path)
            model.load_state_dict(checkpoint['model'])
    model.eval()

    psnr_output_meter = meter.AverageValueMeter()

    image_train_path_moire = "{0}/{1}".format(args.save_prefix, "TEST_Moirefolder")
    image_train_path_clean = "{0}/{1}".format(args.save_prefix, "TEST_Cleanfolder")
    image_train_path_demoire = "{0}/{1}".format(args.save_prefix, "TEST_Demoirefolder")
    if not os.path.exists(image_train_path_moire):      os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean):      os.makedirs(image_train_path_clean)
    if not os.path.exists(image_train_path_demoire):    os.makedirs(image_train_path_demoire)

    for ii ,(moires,clears,labels) in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            moires = moires.to(args.device)
            clears1 = clears
            clears1 = clears1.to(args.device)
            outputs3,outputs2,outputs1 = model(moires)
            outputs1 = torch.clip(outputs1,0,1)

            img_path1 = 'clean.png'
            img_path2 = 'demoire.png'
            img_path3 = 'moire.png'
            vutils.save_image(clears1, img_path1)
            vutils.save_image(outputs1, img_path2)
            vutils.save_image(moires, img_path3)
            exit()

        bs = moires.shape[0]
        for jj in range(bs):
            output, clear, moire, label = outputs1[jj], clears1[jj], moires[jj], labels[jj]
            # if jj ==0 :
            #     print('output',output.shape)
            #     print(clear.shape)
            psnr_output_individual  = PSNR_tensor(output, clear)
            psnr_output_meter.add(psnr_output_individual)
            psnr_output_individual_255  = PSNR_tensor_255(output, clear)
            psnr_output_individual_4095  = PSNR_tensor_4095(output, clear)

            psnr_input_individual   = PSNR_tensor(moire, clear)
            psnr_input_individual_255 = PSNR_tensor_255(moire, clear)
            psnr_input_individual_4095 = PSNR_tensor_4095(moire, clear)


            #8 bit
            img_path1 = "{0}/{1}_A_moire_08bit_{2:.4f}_{3:.4f}.png".format(image_train_path_moire, label, psnr_input_individual,psnr_input_individual_255)
            img_path2 = "{0}/{1}_C_clean.png".format(image_train_path_clean, label)
            img_path3 = "{0}/{1}_B_demoire_08bit_{2:.4f}_{3:.4f}.png".format(image_train_path_demoire, label, psnr_output_individual,psnr_output_individual_255)
            vutils.save_image(moire, img_path1)
            vutils.save_image(clear, img_path2)
            vutils.save_image(output, img_path3)

            #16 bit
            # img_path = "{0}/{1}_B_demoire_12bit_PSNR_{2:.4f}_{3:.4f}.tif".format(image_train_path_demoire, label,psnr_output_individual , psnr_output_individual_4095)
            # img_path2 = "{0}/{1}_A_moire_12bit_PSNR{2:.4f}_{3:.4f}_moire.tif".format(image_train_path_moire, label, psnr_input_individual, psnr_input_individual_4095)
            # img_path3 = "{0}/{1}_C_clean_12bit_.tif".format(image_train_path_clean, label)
            # tensor2img2imwrite_4095(output, img_path)
            # tensor2img2imwrite_4095(moire, img_path2)
            # tensor2img2imwrite_4095(clear, img_path3)

    print('\n1111  =        ',psnr_output_meter.value()[0])

