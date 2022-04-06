import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from Util.util_collections import Time2Str, PSNR_tensor,PSNR_tensor_255, PSNR_tensor_4095,tensor2img2imwrite_4095
from data_loader.dataset import Captured_Moire_dataset_testmode
from torchnet import meter
# from skimage.metrics import peak_signal_noise_ratio
import torchvision.utils as vutils
import time
import math


# from Net.HRDN import get_pose_net as HRDN
##demoire lg2
#4747474747474747474747474747

def test(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    version = "_"+"TEST"
    NET = "_"+"UNet"
    dataset = "_"+"weak_moire_450"
    bit = "_"+"12bit"
    trainsize_testsize = "_"+"train:256_test:2000_1024"
    batchsize = "_" + "batch:" + str(args.batchsize)
    loss = "_" + "MSE"
    detail = "_" + "psnr_0_1" + "12_to_8bit"
    name = version + NET + dataset + bit + trainsize_testsize + batchsize + loss + detail + "_"


    args.save_prefix = args.save_prefix + Time2Str() + name
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)
    print('torch devices = \t', args.device)
    print('save_path     = \t', args.save_prefix)



    Moiredata_test = Captured_Moire_dataset_testmode(args.testmode_path)
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=args.batchsize_testmode,
                                 num_workers=args.num_worker)

    ## model = HRDN일때
    # cfg.merge_from_file("config_jh/cfg.yaml")
    # hrdn_pretrain = None
    # model = hrdn_net(cfg, hrdn_pretrain)  # initweight
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.01  # 0.005 # 0.01
    # )
    ##

    model = nn.DataParallel(model)
    model = model.to(args.device)

    ##dmcnn week
    # path = '/database2/jhkim/PTHfolderIEEEACCESS/220204_15:33_Train_DMCNN_weak_moire_450_8bit_train:256_test:2000_1024_batch:2_MSE_psnr_0_1_/1_weight_folder'
    # name = 'Best_performance_Net_statedict_epoch120_psnr_55.9364dB_.pth'
    #mbcnnweak
    # path = '/database2/jhkim/PTHfolderIEEEACCESS/47pthfolder'
    # name = 'Best_performance_mbcnn_weak_Net_statedict_epoch137_psnr_55.4877dB_.pth'


    #unet
    path = '/database2/jhkim/PTHfolderIEEEACCESS/00_paper_pth'
    # name = 'Best_performance_UNET_weak_augmentation_Net_statedict_epoch465_psnr_58.4514dB_.pth'
    # name = 'Best_performance_UNET_weak_frequency_Net_statedict_epoch387_psnr_63.7360dB_.pth'
    # name = 'Best_performance_UNET_strong_augmentation_Net_statedict_epoch093_psnr_56.3603dB_.pth'
    # name = 'Best_performance_UNET_strong_frequency_Net_statedict_epoch314_psnr_58.2117dB_.pth'
    name = 'Best_performance_UNET_weak_12bit_Net_statedict_epoch165_psnr_61.9173dB_.pth'
    # name = 'Best_performance_UNET_strong_12bit_Net_statedict_epoch361_psnr_59.2908dB_.pth'
    # name = 'Best_performance_CBAM_weak_Net_statedict_epoch324_psnr_65.1887dB_.pth'
    # name = 'Best_performance_CBAM_strong_Net_statedict_epoch331_psnr_61.5481dB_.pth'
    # name = 'Best_performance_MPRB_strong_Net_statedict_epoch348_psnr_63.9051dB_.pth'
    # name = 'Best_performance_MPRB_weak_Net_statedict_epoch365_psnr_67.0190dB_.pth'
    # name = 'Best_performance_UNET_vit_weak_Net_statedict_epoch448_psnr_65.3907dB_.pth'
    # name = 'Best_performance_UNET_vit_strong_Net_statedict_epoch316_psnr_61.4015dB_.pth'
    # name = 'Best_performance_UNET_weak_512_512_Net_statedict_epoch222_psnr_68.0820dB_.pth'
    args.Test_pretrained_path = os.path.join( path , name)
    print(args.Test_pretrained_path)


    ## model = HRDN일때
    ## model = HRDN()


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
            # moires = moires.to(args.device)
            # clears1 = clears
            # clears1 = clears1.to(args.device)
            # outputs1 = model(moires)

################
            ##unet
            moires = moires.cuda()
            clear1 = clears.to(args.device)
            output1 = model(moires)
            # Loss_l1 = criterion_MSE(output1, clear1)
            ##unet

            ##MBCNN
            # moires = torch.stack([moires, moires, moires], dim=1)
            # moires = torch.squeeze(moires, 2)
            # moires = moires.cuda()
            # clear1 = clears.to(args.device)
            # _,_,output1 = model(moires)
            # output1 = torch.mean(output1, 1)
            # output1 = torch.unsqueeze(output1, 1)
            # Loss_l1 = criterion_MSE(output1, clear1)
            ##MBCNN

            ##DMCNN
            # moires = torch.stack([moires, moires, moires], dim=1)
            # moires = torch.squeeze(moires, 2)
            # moires = moires.cuda()
            # clear1 = clears.to(args.device)
            # output1 = model(moires)
            # output1 = torch.mean(output1, 1)
            # output1 = torch.unsqueeze(output1, 1)
            # Loss_l1 = criterion_MSE(output1, clear1)
            ##DMCNN

            ##HRDN
            # moires = torch.stack([moires, moires, moires], dim=1)
            # moires = torch.squeeze(moires, 2)
            # moires = moires.cuda()
            # clear1 = clears.to(args.device)
            # clear1 = torch.stack([clear1, clear1, clear1], dim=1)
            # clear1 = torch.squeeze(clear1, 2)
            # output_list, edge_output_list = model(moires)
            # output1, edge_X = output_list[0], edge_output_list[0]
            #
            # if epoch < 20:
            #     loss_alpha=0.8
            # elif epoch >= 20 and epoch < 40:
            #     loss_alpha = 0.9
            # else:
            #     loss_alpha = 1.0
            #
            # c_loss = criterion_c(output1, clear1)
            # s_loss = criterion_s(edge_X, clear1)
            # Loss_l1 = loss_alpha * c_loss + (1 - loss_alpha) * s_loss
            #
            # output1 = torch.mean(output1, 1)
            # output1 = torch.unsqueeze(output1, 1)
            # clear1 = torch.mean(clear1, 1)
            # clear1 = torch.unsqueeze(clear1, 1)
            ##HRDN


################
        bs = moires.shape[0]
        for jj in range(bs):
            output, clear, moire, label = output1[jj], clear1[jj], moires[jj], labels[jj]

            psnr_output_individual  = PSNR_tensor(output, clear)
            psnr_output_meter.add(psnr_output_individual)
            psnr_output_individual_255  = PSNR_tensor_255(output, clear)
            psnr_output_individual_4095  = PSNR_tensor_4095(output, clear)

            psnr_input_individual   = PSNR_tensor(moire, clear)
            psnr_input_individual_255 = PSNR_tensor_255(moire, clear)
            psnr_input_individual_4095 = PSNR_tensor_4095(moire, clear)


            #8 bit
            img_path1 = "{0}/{1}_A_moire_08bit_{2:.4f}_.png".format(image_train_path_moire, label, psnr_input_individual)
            img_path2 = "{0}/{1}_C_clean.png".format(image_train_path_clean, label)
            img_path3 = "{0}/{1}_B_demoire_psnr_{2:.4f}_.png".format(image_train_path_demoire, label, psnr_output_individual)
            vutils.save_image(moire, img_path1)
            vutils.save_image(clear, img_path2)
            vutils.save_image(output, img_path3)

            # #16 bit
            # img_path = "{0}/{1}_B_demoire_12bit_PSNR_{2:.4f}_{3:.4f}.tif".format(image_train_path_demoire, label,psnr_output_individual , psnr_output_individual_4095)
            # img_path2 = "{0}/{1}_A_moire_12bit_PSNR{2:.4f}_{3:.4f}_moire.tif".format(image_train_path_moire, label, psnr_input_individual, psnr_input_individual_4095)
            # img_path3 = "{0}/{1}_C_clean_12bit_.tif".format(image_train_path_clean, label)
            # tensor2img2imwrite_4095(output, img_path)
            # tensor2img2imwrite_4095(moire, img_path2)
            # tensor2img2imwrite_4095(clear, img_path3)

    print('\npsnr_output_meter  =        ',psnr_output_meter.value()[0])

