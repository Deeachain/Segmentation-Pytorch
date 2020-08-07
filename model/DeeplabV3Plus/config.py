class Configuration():
	def __init__(self):
		self.MODEL_NAME = 'deeplabv3plus'
		self.MODEL_BACKBONE = 'res50_atrous'
		self.MODEL_OUTPUT_STRIDE = 16
		self.MODEL_ASPP_OUTDIM = 256
		self.MODEL_SHORTCUT_DIM = 48
		self.MODEL_SHORTCUT_KERNEL = 1
		self.MODEL_NUM_CLASSES = 3

		self.TRAIN_BN_MOM = 0.0003


		# self.__check()
		# self.__add_path(os.path.join(self.ROOT_DIR, 'lib'))
		
	# def __check(self):
	# 	if not torch.cuda.is_available():
	# 		raise ValueError('config.py: cuda is not avalable')
	# 	if self.TRAIN_GPUS == 0:
	# 		raise ValueError('config.py: the number of GPU is 0')
	# 	#if self.TRAIN_GPUS != torch.cuda.device_count():
	# 	#	raise ValueError('config.py: GPU number is not matched')
	# 	if not os.path.isdir(self.LOG_DIR):
	# 		os.makedirs(self.LOG_DIR)
	# 	if not os.path.isdir(self.MODEL_SAVE_DIR):
	# 		os.makedirs(self.MODEL_SAVE_DIR)
	#
	# def __add_path(self, path):
	# 	if path not in sys.path:
	# 		sys.path.insert(0, path)


cfg = Configuration() 	
