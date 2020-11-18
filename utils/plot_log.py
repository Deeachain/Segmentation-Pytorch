import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def draw_log(args, epoch, mIOU_val_list, lossVal_list):
    f = open(args.savedir + 'log.txt', 'r')
    next(f)
    if args.val_epochs == 1:
        lossTr_list = []
        lossVal_list = []
        for line in f.readlines():
            lossTr_list.append(float(line.strip().split()[2]))
            lossVal_list.append(float(line.strip().split()[3]))
        assert len(lossTr_list) == len(lossVal_list)
        # plt loss
        fig1, ax1 = plt.subplots(figsize=(11, 8))
        ax1.plot(range(1, epoch + 1), lossTr_list, label='Train_loss')
        ax1.plot(range(1, epoch + 1), lossVal_list, label='Val_loss')
        ax1.set_title("Average training loss vs epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current loss")
        ax1.legend()
        plt.savefig(args.savedir + "loss.png")
        plt.close('all')
        plt.clf()
        # plt Miou
        fig2, ax2 = plt.subplots(figsize=(11, 8))
        ax2.plot(range(1, epoch + 1), mIOU_val_list, label="Val IoU")
        ax2.set_title("Average IoU vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current IoU")
        ax2.legend()
        plt.savefig(args.savedir + "mIou.png")
        plt.close('all')
    else:
        # plt loss
        fig1, ax1 = plt.subplots(figsize=(11, 8))
        epoch_list = []
        lossTr_list = []
        for index, line in enumerate(f.readlines()):
            lossTr_list.append(float(line.strip().split()[2]))
            if (index + 1) / args.val_epochs == 0 or index == 0:
                epoch_list.append(int(line.strip().split()[0]))
        ax1.plot(range(1, epoch + 1), lossTr_list, label='Train_loss')
        ax1.plot(epoch_list, lossVal_list, label='Val_loss')
        ax1.set_title("Average loss vs epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Current loss")
        ax1.legend()
        plt.savefig(args.savedir + "loss.png")
        plt.clf()
        # plt Miou
        fig2, ax2 = plt.subplots(figsize=(11, 8))
        ax2.plot(epoch_list, mIOU_val_list, label="Val IoU")
        ax2.set_title("Average IoU vs epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Current IoU")
        ax2.legend()
        plt.savefig(args.savedir + "mIou.png")
        plt.close('all')