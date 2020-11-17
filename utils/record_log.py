import os

class record_log():
    def __init__(self, args):
        self.args = args

    def record_args(self, datas, total_paramters, GLOBAL_SEED):
        with open(self.args.savedir + 'args.txt', 'w') as f:
            f.write('mean:{}\nstd:{}\n'.format(datas['mean'], datas['std']))
            f.write("Parameters: {} Seed: {}\n".format(str(total_paramters), GLOBAL_SEED))
            f.write(str(self.args))

    def initial_logfile(self):
        logFileLoc = self.args.savedir + self.args.logFile
        if os.path.isfile(logFileLoc):
            logger = open(logFileLoc, 'a')
        else:
            logger = open(logFileLoc, 'w')
            logger.write(
                "%s\t%s\t\t%s\t%s\t%s\t%s\n" % ('Epoch', '   lr', 'Loss(Tr)', 'Loss(Val)', 'FWIOU(Val)', 'mIOU(Val)'))
        return logger

    def record_trainVal_log(self, logger, epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, per_class_iu):
        logger.write(
            "%d\t%.6f\t%.4f\t\t%.4f\t\t%0.4f\t\t%0.4f\t\t%s\n" % (
                epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, per_class_iu))
        logger.flush()
        print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\tVal Loss = %.4f\tFWIOU(val) = %.4f\tmIOU(val) = %.4f\tper_class_iu= %s\n" \
              % (epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, str(per_class_iu)))

    def record_train_log(self, logger, epoch, lr, lossTr):
        logger.write("%d\t%.6f\t%.4f\n" % (epoch, lr, lossTr))
        logger.flush()
        print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\n" % (epoch, lr, lossTr))