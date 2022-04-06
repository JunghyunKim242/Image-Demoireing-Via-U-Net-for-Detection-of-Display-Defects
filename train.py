import os
import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader.dataset import Captured_Moire_dataset_train, Captured_Moire_dataset_test
from data_loader.dataset import Captured_Moire_dataset_train_16bit, Captured_Moire_dataset_test_16bit
from torchnet import meter
from Util.util_collections import  Time2Str, PSNR_tensor ,PSNR_tensor_255, PSNR_tensor_4095,tensor2img2imwrite_4095
import torchvision.utils as vutils
##demoire lg2



#################
def train(args, model):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    name = '_Train_UNet_(1,1)_weak_range_adjust_22_10_12bit_finetunning_9_1'


    args.save_prefix = args.save_prefix + Time2Str() + name
    args.pthfolder   = os.path.join( args.save_prefix , '1_weight_folder/')
    if not os.path.exists(args.save_prefix):    os.makedirs(args.save_prefix)
    if not os.path.exists(args.pthfolder)  :    os.makedirs(args.pthfolder)


    print('torch devices = \t', args.device)
    print('save_path     = \t', args.save_prefix)


    #8bit
    # Moiredata_train  = Captured_Moire_dataset_train(args.traindata_path)
    # train_dataloader = DataLoader(Moiredata_train,
    #                               batch_size=args.batchsize,
    #                               shuffle=True,
    #                               num_workers=args.num_worker,
    #                               drop_last=True)
    # Moiredata_test  = Captured_Moire_dataset_test(args.testdata_path)
    # test_dataloader = DataLoader(Moiredata_test,
    #                              batch_size=args.batchsize_test,
    #                              num_workers=args.num_worker)


    ## 16bit
    # Moiredata_train = Captured_Moire_dataset_train_16bit(args.traindata_path)
    # train_dataloader = DataLoader(Moiredata_train,
    #                               batch_size=args.batchsize,
    #                               shuffle=True,
    #                               num_workers=args.num_worker,
    #                               drop_last=True)
    # Moiredata_test = Captured_Moire_dataset_test_16bit(args.testdata_path)
    # test_dataloader = DataLoader(Moiredata_test,
    #                              batch_size=args.batchsize_test,
    #                              num_workers=args.num_worker)

    lr = args.lr
    last_epoch = 0
    optimizer = optim.Adam(params=model.parameters(),
                           lr=lr )

    list_psnr_output = []
    list_loss_output = []

    model = nn.DataParallel(model)
    model = model.cuda()
    model.train()

    if args.Train_pretrained_path:
        checkpoint = torch.load(args.Train_pretrained_path)
        model.load_state_dict(checkpoint['model'])
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        lr = checkpoint['lr']
        list_psnr_output = checkpoint['list_psnr_output']
        list_loss_output = checkpoint['list_loss_output']
        args.bestperformance = checkpoint['bestperformance']
        psnr_output, loss_output1 = test(model, test_dataloader, last_epoch, args)
        print('\nPretrain weight was loaded!')
        print('Pretrained set :PSNR = {:f} \tloss = {:f} \t '.format( psnr_output, loss_output1) )

    # 12bit fine-tunning
    path1 = '/databse4/jhkim/PTHfolder/00_pthfolder/_Train_UNet(1,1)_weak_range_adjust_22_10_12bit_fine_tunning_train_epoch244_psnr_68.2046.pth'
    model.load_state_dict(torch.load(path1), strict=False)
    psnr_output, loss_output1 = test(model, test_dataloader, 51, args)
    print('\nPretrain weight was loaded!')
    print('Pretrained set :PSNR = {:f} \tloss = {:f} \t '.format( psnr_output, loss_output1) )


    criterion_MSE = nn.MSELoss()
    psnr_meter  = meter.AverageValueMeter()
    psnr_meter255  = meter.AverageValueMeter()
    psnr_meter4095  = meter.AverageValueMeter()
    Loss_meter1 = meter.AverageValueMeter()

    for epoch in range(1,args.max_epoch+1):
        if epoch <= last_epoch: #checkpoint로 load했을때 이전 epoch은 건너뛰기 위한 부분입니다.
            continue
        print('\nepoch = {} / {}'.format(epoch , args.max_epoch))

        start = time.time()
        Loss_meter1.reset()
        psnr_meter.reset()
        psnr_meter255.reset()
        psnr_meter4095.reset()

        for  ii, (moires, clears, labels) in tqdm(enumerate(train_dataloader)):
            if ii ==0:
                print('moire.shape',moires.shape)
            moires = moires.cuda()
            clear1 = clears.to(args.device)
            output1 = model(moires)

            Loss_l1                 = criterion_MSE(output1, clear1)
            loss = Loss_l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr = PSNR_tensor(output1, clear1)
            psnr_255 = PSNR_tensor_255(output1, clear1)
            psnr_4095 = PSNR_tensor_4095(output1, clear1)
            psnr_meter.add(psnr)
            psnr_meter255.add(psnr_255)
            psnr_meter4095.add(psnr_4095)
            Loss_meter1.add(loss.item())

        print('training set : \tPSNR = {:f}\t Loss_meter1 = {:f}\t,PSNR255 = {:f}\tPSNR4095 = {:f}  '.format(psnr_meter.value()[0], Loss_meter1.value()[0],psnr_meter255.value()[0],psnr_meter4095.value()[0] ))

        psnr_output, loss_output1 = test(model, test_dataloader, epoch,args)
        print('Test set : \t' + '\033[30m \033[43m' + 'PSNR = {:f}'.format(psnr_output)+'\033[0m'+'\tbest PSNR = {:f}, loss = {:f}'.format(args.bestperformance,loss_output1) )

        if epoch % 50 == 0 :
            print('\033[30m \033[41m' + 'LR was Decreased!!!{:} > {:}    !!!'.format(lr,lr*0.3) + '\033[0m' )
            lr *= 0.3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        list_psnr_output.append( round(psnr_output,7))
        list_loss_output.append( round(loss_output1,7))

        if psnr_output > args.bestperformance  :
            file_name = args.pthfolder + 'Best_performance_{:}_statedict_epoch{:03d}_psnr_{:.4f}.pth'.format(args.name,epoch,psnr_output)
            torch.save(model.state_dict(), file_name)
            print('\033[30m \033[42m' + 'PSNR WAS UPDATED!!!!!!!!!!!!!!!!!!!PSNR += {:f}'.format(psnr_output-args.bestperformance)+'\033[0m')
            args.bestperformance = psnr_output

        if epoch % args.save_every == 0 or epoch == 1 :
            file_name = args.pthfolder + 'Best_performance_{:}_checkpoint_epoch_{:03d}_psnr:{:.4f}_.tar'.format(args.name,epoch,psnr_output)
            if epoch ==1:   file_name = args.pthfolder + 'Best_performance_{:}_epoch_0_initialsetting_.tar'.format(args.name)
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


def test(model, dataloader, epoch, args):
    model.eval()
    criterion_MSE = nn.MSELoss()

    psnr_output_meter = meter.AverageValueMeter()
    loss_meter1 = meter.AverageValueMeter()
    psnr_output_meter.reset()
    loss_meter1.reset()

    image_train_path_demoire = "{0}/epoch_{1:03d}_validation_set_{2}/".format(args.save_prefix, epoch, "demoire")
    image_train_path_moire = "{0}/epoch_{1:03d}_validation_set_{2}/".format(args.save_prefix, 1, "moire")
    image_train_path_clean = "{0}/epoch_{1:03d}_validation_set_{2}/".format(args.save_prefix, 1, "clean")
    if not os.path.exists(image_train_path_moire): os.makedirs(image_train_path_moire)
    if not os.path.exists(image_train_path_clean): os.makedirs(image_train_path_clean)
    if (epoch  % args.save_every == 0 or epoch == 1) and not os.path.exists(image_train_path_demoire) :
        os.makedirs(image_train_path_demoire)

    for ii, (val_moires, val_clears, labels) in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            val_moires = val_moires.to(args.device)
            output1 = model(val_moires)
            val_clears = val_clears.to(args.device)

        loss = criterion_MSE(output1, val_clears)
        loss_meter1.add(loss.item())

        batch = val_moires.shape[0]
        for jj in range(batch):
            output, clear, moire, label = output1[jj], val_clears[jj], val_moires[jj], labels[jj]
            psnr_output_individual  = PSNR_tensor(output, clear)
            psnr_output_meter.add(psnr_output_individual)

            psnr_output_individual_255  = PSNR_tensor_255(output, clear)
            psnr_output_individual_4095  = PSNR_tensor_4095(output, clear)

            if epoch % args.save_every == 0 :
                img_path = "{0}/{1}_B_epoch:{2:04d}_demoire_PSNR:{3:.4f}_{4:.4f}.png".format(image_train_path_demoire, label,epoch ,psnr_output_individual, psnr_output_individual_255)
                vutils.save_image(output, img_path)
                ############16bit
                img_path = "{0}/{1}_B_epoch:{2:04d}_demoire_PSNR_12bit:{3:.4f}.tif".format(image_train_path_demoire, label,epoch , psnr_output_individual_4095)
                tensor2img2imwrite_4095(output, img_path)

            if epoch == 1 :
                psnr_input_individual = PSNR_tensor(moire, clear)
                psnr_input_individual_255 = PSNR_tensor_255(moire, clear)
                psnr_input_individual_4095 = PSNR_tensor_4095(moire, clear)

                img_path  = "{0}/{1}_B_epoch:{2:04d}_demoire_PSNR:{3:.4f}_{4:.4f}.png".format(image_train_path_demoire, label, epoch, psnr_output_individual,psnr_output_individual_255)
                img_path2 = "{0}/{1}_A_moire_PSNR:{2:.4f},{3:.4f}_moire.png".format( image_train_path_moire, label, psnr_input_individual, psnr_input_individual_255)
                img_path3 = "{0}/{1}_C_clean_.png".format(         image_train_path_clean, label)
                vutils.save_image(output, img_path)
                vutils.save_image(moire, img_path2)
                vutils.save_image(clear, img_path3)

                ############16bit
                img_path2 = "{0}/{1}_A_moire_12bitPSNR{2:.4f}_{3:.4f}_moire.tif".format(image_train_path_moire, label, psnr_input_individual, psnr_input_individual_4095)
                img_path3 = "{0}/{1}_C_clean_12bit_.tif".format(image_train_path_clean, label)
                tensor2img2imwrite_4095(moire, img_path2)
                tensor2img2imwrite_4095(clear, img_path3)

    model.train()
    return psnr_output_meter.value()[0], loss_meter1.value()[0]


