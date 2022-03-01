import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import sys

# Run directly from train.py
from pathlib import Path
sys.path.append(str(Path(os.path.join(os.getcwd(), 'gmcnn'))))
#sys.path.append("c:\\Users\\Jegern\\Masteroppgave\\gmcnn")

from tensorboardX import SummaryWriter
#from data.data import InpaintingDataset, ToTensor
#from data.dataset import KvasirDataset, ToTensor
from data.dataset import KvasirDataset, ToTensor
from models.inpainting_gmcnn.net import InpaintingModel_GMCNN
from configs.gmcnn.train_options import TrainOptions
from models.inpainting_gmcnn.utils import getLatest


config = TrainOptions().parse()
print(config)
print('loading data..')
print(config.root_path)
dataset = KvasirDataset(root_path=config.root_path, transform=transforms.Compose([ToTensor()]), args=config)

dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)
print('data loaded..')

print('configuring model..')
ourModel = InpaintingModel_GMCNN(in_channels=4, opt=config)
ourModel.print_networks()
print(config.load_model_dir)
if config.load_model_dir != '':
    print('Loading pretrained model from {}'.format(config.load_model_dir))
    ourModel.load_networks(getLatest(os.path.join(config.load_model_dir, '*.pth')))
    print('Loading done.')
# ourModel = torch.nn.DataParallel(ourModel).cuda()
print('model setting up..')
print('training initializing..')
writer = SummaryWriter(log_dir=config.model_folder)
cnt = 0
for epoch in range(config.epochs):

    for i, data in enumerate(dataloader):
        gt = data['gt'].cuda()
        
        if "custom" in config.mask_type:
            mask_data = data['mask'].cuda()
            gt = gt / 127.5 - 1
            data_in = {'gt': gt, 'mask': mask_data}
        else:

            # normalize to values between -1 and 1
            gt = gt / 127.5 - 1
            data_in = {'gt': gt}
        
        ourModel.setInput(data_in)
        ourModel.optimize_parameters()

        if (i+1) % config.viz_steps == 0:
            ret_loss = ourModel.get_current_losses()
            if config.pretrain_network is False:
                print(
                    '[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f, adv: %.4f, mrf: %.4f), D_loss: %.4f'
                    % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae'],
                       ret_loss['G_loss_adv'], ret_loss['G_loss_mrf'], ret_loss['D_loss']))
                writer.add_scalar('adv_loss', ret_loss['G_loss_adv'], cnt)
                writer.add_scalar('D_loss', ret_loss['D_loss'], cnt)
                writer.add_scalar('G_mrf_loss', ret_loss['G_loss_mrf'], cnt)
            else:
                print('[%d, %5d] G_loss: %.4f (rec: %.4f, ae: %.4f)'
                      % (epoch + 1, i + 1, ret_loss['G_loss'], ret_loss['G_loss_rec'], ret_loss['G_loss_ae']))

            writer.add_scalar('G_loss', ret_loss['G_loss'], cnt)
            writer.add_scalar('reconstruction_loss', ret_loss['G_loss_rec'], cnt)
            writer.add_scalar('autoencoder_loss', ret_loss['G_loss_ae'], cnt)

            images = ourModel.get_current_visuals_tensor()
            im_completed = vutils.make_grid(images['completed'], normalize=True, scale_each=True)
            im_input = vutils.make_grid(images['input'], normalize=True, scale_each=True)
            im_gt = vutils.make_grid(images['gt'], normalize=True, scale_each=True)
            writer.add_image('gt', im_gt, cnt)
            writer.add_image('input', im_input, cnt)
            writer.add_image('completed', im_completed, cnt)
            if (i+1) % config.train_spe == 0:
                print('saving model ..')
                ourModel.save_networks(epoch+1)
        cnt += 1
    ourModel.save_networks(epoch+1)

writer.export_scalars_to_json(os.path.join(config.model_folder, 'GMCNN_scalars.json'))
writer.close()