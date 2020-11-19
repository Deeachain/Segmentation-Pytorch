from prettytable import PrettyTable

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
        logger = open(logFileLoc, 'w')
        logger.write(
            "%s\t%s\t\t%s\t%s\t%s\t%s\n" % ('Epoch', '   lr', 'Loss(Tr)', 'Loss(Val)', 'FWIOU(Val)', 'mIOU(Val)'))
        return logger

    def resume_logfile(self):
        logFileLoc = self.args.savedir + self.args.logFile
        logger_recored = open(logFileLoc, 'r')
        next(logger_recored)
        lines = logger_recored.readlines()
        logger_recored.close()
        logger = open(logFileLoc, 'a+')
        return logger, lines

    def record_trainVal_log(self, logger, epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, PerMiou_set, class_dict_df):
        logger.write(
            "%d\t%.6f\t%.4f\t\t%.4f\t\t%0.4f\t\t%0.4f\t\t%s\n" % (
                epoch, lr, lossTr, val_loss, FWIoU, mIOU_val, PerMiou_set))
        logger.flush()
        print("Epoch %d\tlr= %.6f\tTrain Loss = %.4f\tVal Loss = %.4f\tFWIOU(val) = %.4f" \
              % (epoch, lr, lossTr, val_loss, FWIoU))

        t = PrettyTable(['label_index', 'class_name', 'class_iou'])
        for index in range(class_dict_df.shape[0]):
            t.add_row([class_dict_df['label_index'][index], class_dict_df['class_name'][index], PerMiou_set[index]])
        print(t.get_string(title="Miou is {}".format(mIOU_val)))

    def record_train_log(self, logger, epoch, lr, lossTr):
        logger.write("%d\t%.6f\t%.4f\n" % (epoch, lr, lossTr))
        logger.flush()
        print("Epoch  %d\tlr= %.6f\tTrain Loss = %.4f\n" % (epoch, lr, lossTr))