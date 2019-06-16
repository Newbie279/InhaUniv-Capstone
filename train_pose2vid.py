import os
import numpy as np
import torch
import time
import sys
from collections import OrderedDict
from torch.autograd import Variable
from pathlib import Path
import warnings

debug = False
target_model_num = 3

warnings.filterwarnings('ignore')
mainpath = os.getcwd()
pix2pixhd_dir = Path(mainpath+'/src/pix2pixHD/')
sys.path.append(str(pix2pixhd_dir))

import src.pix2pixHD.models.pix2pixHD_model as p2p
from src.pix2pixHD.data.data_loader import CreateDataLoader
from src.pix2pixHD.models.models import create_model
import util.util as util
from util.visualizer import Visualizer
import src.config.train_opt as opt

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.benchmark = True



def make_vector_channel(y, n_dims, shape=(512, 512)):
    def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
        scatter_dim = len(y.size())
        y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
        zeros = torch.zeros(*y.size(), n_dims).type(dtype)

        return zeros.scatter(scatter_dim, y_tensor, 1)

    onehot = _to_one_hot(y, n_dims)

    x_repeat = shape[1] // n_dims + 1
    return onehot.repeat(shape[0], x_repeat)[:,:512]

def exec_train(mv_name, end_epoch):

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    start_epoch, epoch_iter = 1, 0
    total_steps = (start_epoch - 1) * dataset_size + epoch_iter
    display_delta = total_steps % opt.display_freq
    print_delta = total_steps % opt.print_freq
    save_delta = total_steps % opt.save_latest_freq

    model = create_model(opt)
    model = model.cuda()
    visualizer = Visualizer(opt)

    flag = True

    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        if epoch != start_epoch:
            epoch_iter = epoch_iter % dataset_size

        for i, data in enumerate(dataset, start=epoch_iter):
            iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            # whether to collect output images
            save_fake = total_steps % opt.display_freq == display_delta

            tmp_num=3
            if int(str(data['path']).split('/')[6].split('.')[0]) > 45776:
                tmp_num=4
            else:
                tmp_num=3

            onehot_map = Variable(make_vector_channel(torch.tensor([tmp_num]), 10).unsqueeze(0).unsqueeze(0))
            concated_label = torch.cat((Variable(data['label']), onehot_map.cpu()), dim=1)
            ############## Forward Pass ######################
            losses, generated = model(Variable(concated_label), Variable(data['inst']),
                                      Variable(data['image']), Variable(data['feat']),  infer=save_fake)

            #inst and feature doesn't pass through this model
            #must make dataset for inst and feature

            # sum per device losses
            losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
            loss_dict = dict(zip(model.loss_names, losses))

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)

            ############### Backward Pass ####################
            # update generator weights
            model.optimizer_G.zero_grad()
            loss_G.backward(retain_graph=False)
            model.optimizer_G.step()

            # update discriminator weights
            model.optimizer_D.zero_grad()
            loss_D.backward(retain_graph=False)
            model.optimizer_D.step()


            ############## Display results and errors ##########
            ### print out errors
            if total_steps % opt.print_freq == print_delta:

                for k,v in loss_dict.items():
                    print("k : , v : ",k,v)
                print(loss_dict.items())
                #i fixed this for loops
                #errors = {k: v[0] if not isinstance(v, int) else v for k, v in loss_dict.items()}
                errors = {k: v.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if save_fake:
                visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                                       ('synthesized_image', util.tensor2im(generated.data[0])),
                                       ('real_image', util.tensor2im(data['image'][0]))])
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### save latest model
            if total_steps % opt.save_latest_freq == save_delta:
                print(opt.dataroot)
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                model.save('latest')
                np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

            if epoch_iter >= dataset_size:
                break

            if(i%100==0):
                print("index in current epoch "+str(i))
        # end of epoch
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
            np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
#
        ### instead of only training the local enhancer, train the entire network after certain iterations
        if (opt.niter_fix_global != 0) and (epoch == opt.niter_fix_global):
            model.update_fixed_params()

        ### linearly decay learning rate after certain iterations
        if epoch > opt.niter:
            model.update_learning_rate()


        if not flag:
            break

    torch.cuda.empty_cache()


opt.change_dataroot(3)
exec_train(3,40)