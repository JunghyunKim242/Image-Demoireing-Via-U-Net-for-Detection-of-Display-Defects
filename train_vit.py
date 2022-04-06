import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader.dataset import Captured_Moire_dataset_train_MHA, Captured_Moire_dataset_test_MHA
from torchnet import meter
from Util.util_collections import  Time2Str, PSNR_tensor ,PSNR_tensor_255, PSNR_tensor_4095,tensor2img2imwrite_4095
import torchvision.utils as vutils
from torchvision import transforms
from config_jh import cfg
from Net.hrdn import get_pose_net as hrdn_net

# from Net.HRDN import get_pose_net as HRDN
from Net.LossNet import L1_Sobel_Loss ,L1_Charbonnier_loss


###########################
def train_IEEEACCESS1(args, model):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    version = "_" + "Train"
    NET = "_" + "UNet_with_CBAM_skip_4_3_2_1_all_finetunning"
    dataset = "_" + "strong_moire_450"
    bit = "_" + "8bit"
    trainsize_testsize = "_"+"train:256_test:2000_1024"
    batchsize = "_" + "batch:" + str(args.batchsize)
    loss = "_"+"MSE"
    detail = "_" + "psnr_0_1" + "lr_1_10"
    name = version + NET + dataset + bit + trainsize_testsize + batchsize + loss + detail + "_"


    args.save_prefix = args.save_prefix + Time2Str() + name
    args.pthfolder   = os.path.join( args.save_prefix , '1_weight_folder/')
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)
    if not os.path.exists(args.pthfolder)  :    os.makedirs(args.pthfolder)
    print('torch devices = \t', args.device)
    print('save_path     = \t', args.save_prefix)


    Moiredata_train = Captured_Moire_dataset_train_MHA(args.traindata_path)
    train_dataloader = DataLoader(Moiredata_train,
                                  batch_size=args.batchsize,
                                  shuffle=True,
                                  num_workers=args.num_worker,
                                  drop_last=True)
    Moiredata_test = Captured_Moire_dataset_test_MHA(args.testdata_path)
    test_dataloader = DataLoader(Moiredata_test,
                                 batch_size=args.batchsize_test,
                                 num_workers=args.num_worker)

    # args.lr=1e-2
    lr = args.lr
    ###############
    ###############
    lr = lr*0.1
    ###############
    ###############
    ###############
    last_epoch = 0
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    list_psnr_output = []
    list_loss_output = []

    model = nn.DataParallel(model)
    model = model.cuda()


    folder = '/databse4/jhkim/PTHfolderIEEEACCESS/00_IEEE/'
    name = '02_Best_performance_cbam_4_3_2_1_Net_statedict_epoch472_psnr_58.7873dB_.pth'
    args.Train_pretrained_path = folder + name
    if args.Train_pretrained_path:
        # checkpoint = torch.load(args.Train_pretrained_path)
        # model.load_state_dict(checkpoint['model'])
        # last_epoch = checkpoint["epoch"]
        # optimizer_state = checkpoint["optimizer"]
        # optimizer.load_state_dict(optimizer_state)
        # lr = checkpoint['lr']
        # list_psnr_output = checkpoint['list_psnr_output']
        # list_loss_output = checkpoint['list_loss_output']
        # args.bestperformance = checkpoint['bestperformance']

        # statedict
        model.load_state_dict(torch.load(args.Train_pretrained_path))
        args.bestperformance = 0
        psnr_output, loss_output1,_ = test(model, test_dataloader, last_epoch, args)
        print('\nPretrain',True)
        print('\nPretrain weight was loaded!',args.Train_pretrained_path)
        print('Pretrained set :PSNR = {:f} \tloss = {:f} \t '.format( psnr_output, loss_output1) )
        print('LR = ',lr , 'last_epoch = ',last_epoch,'args.bestperformance = ',args.bestperformance)


    ##12bit
    # path = "/database2/jhkim/PTHfolderIEEEACCESS/47pthfolder/_Train_UNet(1,1)_strong_moire_450_8bit_train:256_test:2000_1024_batch:2_MSE_psnr_0_1_psnr_57.3901dB_.pth"
    # model.load_state_dict(torch.load(path))


    criterion_MSE = nn.MSELoss()
    criterion_L1 = nn.L1Loss()


    psnr_meter  = meter.AverageValueMeter()
    psnr_meter255  = meter.AverageValueMeter()
    psnr_meter4095  = meter.AverageValueMeter()
    Loss_meter1 = meter.AverageValueMeter()

    for epoch in range(1,args.max_epoch+1):
        if epoch <= last_epoch:
            continue
        print('\nepoch = {} / {}'.format(epoch , args.max_epoch))

        start = time.time()
        Loss_meter1.reset()
        psnr_meter.reset()
        psnr_meter255.reset()
        psnr_meter4095.reset()

        for  ii, (moires, clears, labels) in tqdm(enumerate(train_dataloader)):
            model.train()
            if ii == 0:
                print('moire.shape ',moires.shape,"\n path=",args.save_prefix)
            ##unet
            moires = moires.cuda()
            clear1 = clears.to(args.device)
            output1 = model(moires)
            Loss_l1                 = criterion_MSE(output1, clear1)

            ## frequency loss
            gt_fre      = torch.fft.fft(clear1)
            output_fre  = torch.fft.fft(output1)
            loss_fre                = criterion_L1(gt_fre, output_fre)
            Loss_l1 = Loss_l1 + loss_fre
            #unet


            optimizer.zero_grad()
            Loss_l1.backward()
            optimizer.step()


            psnr = PSNR_tensor(output1, clear1)
            psnr_255 = PSNR_tensor_255(output1, clear1)
            psnr_4095 = PSNR_tensor_4095(output1, clear1)
            psnr_meter.add(psnr)
            psnr_meter255.add(psnr_255)
            psnr_meter4095.add(psnr_4095)
            Loss_meter1.add(Loss_l1.item())


        print('training set : \tPSNR0to1 = {:f},\tPSNR255 = {:f},\t Loss_meter1 = {:f},\tPSNR4095 = {:f}  '.format(psnr_meter.value()[0],psnr_meter255.value()[0], Loss_meter1.value()[0],psnr_meter4095.value()[0] ))
        psnr_output, loss_output1,psnr_output255 = test(model, test_dataloader, epoch, args)
        # print('\033[30m \033[43m'+ 'Test set : \t PSNR0_1 = {:f}_'.format(psnr_output) + '\033[0m' + '\tbest PSNR = {:f}, loss = {:f}, '.format(args.bestperformance, loss_output1 ) )
        print('\033[30m \033[43m'+ 'Test set : \t PSNR0_1 = {:f}_PSNR255 = {:f}'.format(psnr_output, psnr_output255) + '\033[0m' + '\tbest PSNR = {:f}, loss = {:f}, '.format(args.bestperformance, loss_output1 ) )

        if epoch % 50 == 0 :
            print('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:} !!!'.format(lr,lr*0.3) + '\033[0m' )
            lr *= 0.3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        list_psnr_output.append( round(psnr_output,7))
        list_loss_output.append( round(loss_output1,7))


        if psnr_output > args.bestperformance  :
            file_name = args.pthfolder + '02_Best_performance_{:}_statedict_epoch{:03d}_psnr_{:.4f}dB_.pth'.format(args.name,epoch,psnr_output)
            torch.save(model.state_dict(), file_name)
            print('\033[30m \033[42m' + 'PSNR255 WAS UPDATED!!!!!!!!!!!!!!!!!!!PSNR += {:f}'.format(psnr_output-args.bestperformance)+'\033[0m')
            if epoch > 5:
                psnr_output, loss_output1, psnr_output255  = test(model, test_dataloader, epoch, args, True)
            args.bestperformance = psnr_output


        if epoch % args.save_every == 0 or epoch == 1 :
            file_name = args.pthfolder + '01_Last_epoch_{:}_checkpoint_.tar'.format( args.name )
            if epoch ==1:   file_name = args.pthfolder + '01_first_epoch_{:}_initial_setting_.tar'.format(args.name)
            checkpoint = {  'epoch': epoch ,
                            "optimizer": optimizer.state_dict(),
                            "model": model.state_dict(),
                            "lr": lr,
                            "list_psnr_output": list_psnr_output,
                            "list_loss_output": list_loss_output,
                            'bestperformance':args.bestperformance,
                            }
            torch.save(checkpoint, file_name)

            with open(args.save_prefix + "/1_PSNR.txt", 'w') as f:
                f.write("psnr_output: {:}\n".format( list_psnr_output ))
            with open(args.save_prefix + "/1_Loss.txt", 'w') as f:
                f.write("loss_output: {:}\n".format( list_loss_output ))

        print('1 epoch spends:{:.2f}sec\t remain {:2d}:{:2d} hours'.format(
            (time.time() - start),
            int((args.max_epoch - epoch) * (time.time() - start) // 3600) ,
            int((args.max_epoch - epoch) * (time.time() - start) % 3600 / 60 ) ))
    return "Training Finished!"


def test(model, dataloader, epoch, args,save = False):
    model.eval()
    criterion_MSE = nn.MSELoss()
    criterion_L1 = nn.L1Loss()


    psnr_output_meter       = meter.AverageValueMeter()
    psnr_output_meter.reset()
    psnr_output_meter_255   = meter.AverageValueMeter()
    psnr_output_meter_255.reset()
    loss_meter1             = meter.AverageValueMeter()
    loss_meter1.reset()


    image_train_path_demoire = "{0}/epoch_{1:03d}_validation_set_{2}/".format(args.save_prefix, epoch, "demoire")
    image_train_path_moire = "{0}/epoch_{1:03d}_validation_set_{2}/".format(args.save_prefix, 1, "moire")
    image_train_path_clean = "{0}/epoch_{1:03d}_validation_set_{2}/".format(args.save_prefix, 1, "clean")
    if not os.path.exists(image_train_path_moire): os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean): os.makedirs(image_train_path_clean)



    for ii, (val_moires, val_clears, labels) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            #unet
            val_moires = val_moires.to(args.device)
            output1 = model(val_moires)
            val_clears = val_clears.to(args.device)
            Loss_l1 = criterion_MSE(output1, val_clears)

        # frequency loss
        gt_fre      = torch.fft.fft(val_clears)
        output_fre  = torch.fft.fft(output1)
        loss_fre                = criterion_L1(gt_fre, output_fre)
        Loss_l1 = Loss_l1 + loss_fre
        #unet

        loss_meter1.add(Loss_l1.item())

        batch = val_moires.shape[0]
        for jj in range(batch):
            output, clear, moire, label = output1[jj], val_clears[jj], val_moires[jj], labels[jj]
            psnr_output_individual      = PSNR_tensor(output, clear)
            psnr_output_individual_255  = PSNR_tensor_255(output, clear)
            psnr_output_meter.add(psnr_output_individual)
            psnr_output_meter_255.add(psnr_output_individual_255)

            if epoch % args.save_every == 0 or save :
                if not os.path.exists(image_train_path_demoire):
                    os.makedirs(image_train_path_demoire)
                img_path = "{0}/{1}_B_epoch:{2:04d}_demoire_PSNR:{3:.4f}_.png".format(image_train_path_demoire, label,epoch ,psnr_output_individual)
                vutils.save_image(output, img_path)

                ############16bit
                # img_path = "{0}/{1}_B_epoch:{2:04d}_demoire_PSNR_12bit:{3:.4f}.tif".format(image_train_path_demoire, label,epoch , psnr_output_individual_4095)
                # tensor2img2imwrite_4095(output, img_path)

            if epoch == 1 :
                if not os.path.exists(image_train_path_demoire):
                    os.makedirs(image_train_path_demoire)
                psnr_input_individual = PSNR_tensor(moire, clear)
                psnr_input_individual_255 = PSNR_tensor_255(moire, clear)
                psnr_input_individual_4095 = PSNR_tensor_4095(moire, clear)

                img_path  = "{0}/{1}_B_epoch:{2:04d}_demoire_PSNR:{3:.4f}_.png".format(image_train_path_demoire, label, epoch, psnr_output_individual)
                img_path2 = "{0}/{1}_A_moire_PSNR:{2:.4f}_.png".format( image_train_path_moire, label, psnr_input_individual)
                img_path3 = "{0}/{1}_C_clean_.png".format(         image_train_path_clean, label)
                vutils.save_image(output, img_path)
                vutils.save_image(moire, img_path2)
                vutils.save_image(clear, img_path3)

                ############16bit
                # img_path2 = "{0}/{1}_A_moire_12bitPSNR{2:.4f}_{3:.4f}_moire.tif".format(image_train_path_moire, label, psnr_input_individual, psnr_input_individual_4095)
                # img_path3 = "{0}/{1}_C_clean_12bit_.tif".format(image_train_path_clean, label)
                # tensor2img2imwrite_4095(moire, img_path2)
                # tensor2img2imwrite_4095(clear, img_path3)


    model.train()
    return psnr_output_meter.value()[0], loss_meter1.value()[0], psnr_output_meter_255.value()[0]






######################################################train

    ## model = HRDN일때
    # cfg.merge_from_file("config_jh/cfg.yaml")
    # path = '/database2/jhkim/PTHfolderIEEEACCESS/47pthfolder/'
    # name = 'Best_performance_hrdn_strong_Net_statedict_epoch159_psnr_49.8839dB_.pth'
    # hrdn_pretrain = None # os.path.join(path, name)
    # model = hrdn_net(cfg, hrdn_pretrain)
    # optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=0.01  # 0.005 # 0.01
    # )

##MBCNN
# moires = torch.stack([moires, moires, moires], dim=1)
# moires = torch.squeeze(moires, 2)
# moires = moires.cuda()
# clear1 = clears.to(args.device)
# _,_,output1 = model(moires)
# output1 = torch.mean(output1, 1)
# output1 = torch.unsqueeze(output1, 1)
# Loss_l1 = criterion_MSE(output1, clear1)


##DMCNN
# moires = torch.stack([moires, moires, moires], dim=1)
# moires = torch.squeeze(moires, 2)
# moires = moires.cuda()
# clear1 = clears.to(args.device)
# output1 = model(moires)
# output1 = torch.mean(output1, 1)
# output1 = torch.unsqueeze(output1, 1)
# Loss_l1 = criterion_MSE(output1, clear1)


##HRDN
# moires = torch.stack([moires, moires, moires], dim=1)
# moires = torch.squeeze(moires, 2)
# moires = moires.cuda()
# clear1 = clears.to(args.device)
# clear1 = torch.stack([clear1, clear1, clear1], dim=1)
# clear1 = torch.squeeze(clear1, 2)
# output_list, edge_output_list = model(moires)
# output1, edge_X = output_list[0], edge_output_list[0]
# if epoch < 20:
#     loss_alpha=0.8
# elif epoch >= 20 and epoch < 40:
#     loss_alpha = 0.9
# else:
#     loss_alpha = 1.0
#
# c_loss = criterion_c(output1, clear1)
# s_loss = criterion_s(edge_X, clear1)
#
# Loss_l1 = loss_alpha * c_loss + (1 - loss_alpha) * s_loss
#
# output1 = torch.mean(output1, 1)
# output1 = torch.unsqueeze(output1, 1)
# clear1 = torch.mean(clear1, 1)
# clear1 = torch.unsqueeze(clear1, 1)
##HRDN
# loss = Loss_l1








######################################################test


##mbcnn
# val_moires = torch.stack([val_moires, val_moires, val_moires], dim=1)
# val_moires = torch.squeeze(val_moires, 2)
# val_moires = val_moires.cuda()
# val_clears = val_clears.to(args.device)
# _,_,output1 = model(val_moires)
# output1 = torch.mean(output1, 1)
# output1 = torch.unsqueeze(output1, 1)
# Loss_l1 = criterion_MSE(output1, val_clears)
##mbcnn


##DMCNN
# val_moires = torch.stack([val_moires, val_moires, val_moires], dim=1)
# val_moires = torch.squeeze(val_moires, 2)
# val_moires = val_moires.cuda()
# val_clears = val_clears.to(args.device)
# output1 = model(val_moires)
# output1 = torch.mean(output1, 1)
# output1 = torch.unsqueeze(output1, 1)
# Loss_l1 = criterion_MSE(output1, val_clears)
##DMCNN


##HRDN
# val_moires = torch.stack([val_moires, val_moires, val_moires], dim=1)
# val_moires = torch.squeeze(val_moires, 2)
# val_moires = val_moires.cuda()
# val_clears = val_clears.to(args.device)
# val_clears = torch.stack([val_clears, val_clears, val_clears], dim=1)
# val_clears = torch.squeeze(val_clears, 2)
# output_list, edge_output_list = model(val_moires)
# output1, edge_X = output_list[0], edge_output_list[0]
#
# if epoch < 20:
#     loss_alpha = 0.8
# elif epoch >= 20 and epoch < 40:
#     loss_alpha = 0.9
# else:
#     loss_alpha = 1.0
#
# c_loss = criterion_c(output1, val_clears)
# s_loss = criterion_s(edge_X, val_clears)
# Loss_l1 = loss_alpha * c_loss + (1 - loss_alpha) * s_loss
# # Loss_l1 = criterion_MSE(output1, val_clears)
#
#
# output1 = torch.mean(output1, 1)
# output1 = torch.unsqueeze(output1, 1)
# val_clears = torch.mean(val_clears, 1)
# val_clears = torch.unsqueeze(val_clears, 1)
##HRDN
