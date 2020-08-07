#--model SegNet    ENet    ESPNet_v2   UNet    PSPNet50
#--optim 'sgd', 'adam', 'radam', 'ranger'

# small model
#python train.py --max_epochs 100 --batch_size 16 --model ENet --dataset paris --optim adam --lr 0.001
#python train.py --max_epochs 100 --batch_size 16 --model ESPNet_v2 --dataset paris --optim sgd --lr 0.001

# large model
#python train.py --max_epochs 100 --batch_size 4 --model UNet --dataset paris --optim sgd --lr 0.01
#python train.py --max_epochs 100 --batch_size 4 --model PSPNet50 --dataset paris --optim sgd --lr 0.001
python train.py --max_epochs 100 --batch_size 4 --model Deeplabv3plus --dataset paris --optim sgd --lr 0.001