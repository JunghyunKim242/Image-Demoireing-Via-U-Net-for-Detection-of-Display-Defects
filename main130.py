import argparse

#train
from train import train
from train_mbcnn_captured import train_MBCNN_captured
from train_mbcnn_AIM import train_MBCNN_AIM
from train_vit import train_IEEEACCESS1

#test
from test_mbcnn import test_MBCNN
from test import test

#Network
from Net.UNet_all import *
from Net.MBCNN import MBCNN1
from Net.DMCNN import MoireCNN as DMCNN
from Net.U_square_net import U2NET


#130130130130130130130130

#초기 hyper parameter를 설정하기 위한 단계입니다.
parser = argparse.ArgumentParser()
parser.add_argument('--traindata_path', type=str,
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_weak/train_450',    help='vit_patches_size, default is 16')
                    default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_strong/train_450',    help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_weak_12bit_2/train_16/',    help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_strong_12bit/train_16',    help='vit_patches_size, default is 16')


parser.add_argument('--testdata_path', type=str,
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_weak/test_50', help='vit_patches_size, default is 16')
                    default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_strong/test_50',    help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_weak_12bit_2/test_4/',    help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_strong_12bit/test_4',    help='vit_patches_size, default is 16')


parser.add_argument('--testmode_path', type=str,
                    default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_weak/test_50', help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_strong/test_50',help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_weak_12bit_2/train_16',help='vit_patches_size, default is 16')
                    # default= '/databse4/jhkim/DataSet/9_mura,iphonemoire,automodel/LG_Demoire_dataset_strong_12bit/train_16/',help='vit_patches_size, default is 16')


parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')

parser.add_argument('--num_worker', type=int, default=12,
                    help='number of workers')

parser.add_argument('--batchsize', type=int,default= 2,
                    help='mini batch size')

parser.add_argument('--batchsize_test', type=int,default = 1,
                    help='mini batch size')

parser.add_argument('--batchsize_testmode', type=int,default= 1,
                    help='mini batch size')

parser.add_argument('--max_epoch', type=int, default=500,
                    help='number of max_epoch')

parser.add_argument('--save_every', type=int,default= 20,
                    help='saving period for pretrained weight ')

parser.add_argument('--name', type=str,default='Net',
                    help='name for this experiment rate')

parser.add_argument('--pthfolder', type=str, default = 'pthfolder path was not configured' ,
                    help='saving folder directory')

parser.add_argument('--device', type=str, default='cuda or cpu',
                    help='device, define it first!!')

parser.add_argument('--save_prefix', type=str, default='/databse4/jhkim/PTHfolderIEEEACCESS/',
                    help='saving folder directory')

parser.add_argument('--bestperformance', type=float, default=0.,
                    help='saving folder directory')

parser.add_argument('--Train_pretrained_path', type=str, default = None, # '/databse4/jhkim/PTHfolder/220107_14:54_Train_UNet_with_CBAM_(3,3)_week_LG_Production Engineering Research Institude_/1_pretrained_weight_folder/Best_performance_Net_epoch_450_checkpoint.tar',
                    help='saving folder directory')

parser.add_argument('--Test_pretrained_path', type=str, default = None, # '/databse4/jhkim/PTHfolderIEEEACCESS/00_pthfolder/_Train_UNet(1,1)_week_moire_450_8bit_train:512_test:2000_1000_batch:2_MSE_psnr_65.5705_psnr255_49.0649.pth',
                    help='saving folder directory')

args = parser.parse_args()
if __name__ == "__main__":
    ############ ACCESS
    # net = UNet(1,1)
    # net = UNet_with_MultiHeadAttention(1,1)
    # net = UNet_with_MultiHeadAttention_1skipconnection(1,1)
    # net = UNet_with_MPRB(1,1)
    # net = UNet_with_CBAM_12skipconnection(1,1)
    # net = UNet_with_CBAM_1skip(1,1)


    # net = UNet(1,1)
    # net = UNet_with_MPRB_skip_4(1,1)
    # net = UNet_with_MPRB_skip_4_1(1,1)
    # net = UNet_with_MPRB_skip_4_3_2_1(1,1)
    # net = UNet_with_CBAM_skip_4(1,1)
    # net = UNet_with_CBAM_skip_4_1(1,1)
    # net = UNet_with_CBAM_xxx(1,1)
    # net = UNet_with_CBAM_raw_sig(1,1)
    net = UNet_with_CBAM_skip_4_3_2_1(1,1)
    # net = MBCNN1(64)
    # net = DMCNN()


    train_IEEEACCESS1(args, net)
    # train_MBCNN_captured(args, net)
    # test_MBCNN(args, net)
    # test(args, net)


    ################ AIM dataset
    # net = MBCNN1(64)
    # train_MBCNN_AIM(args, net)