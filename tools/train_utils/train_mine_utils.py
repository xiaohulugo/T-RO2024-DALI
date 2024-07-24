import glob
import os
import numpy as np
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import pickle
from visual_utils import visualize_utils


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'metadata', 'calib', 'image_shape']:
            continue
        batch_dict[key] = torch.from_numpy(val).float().cuda()


def train_one_epoch_st(model, optimizer, source_loader, target_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, tb_log=None, leave_pbar=False, weights=[1.0,1.0]):
    if source_loader is not None:
        dataloader_iter_s = iter(source_loader)
    if target_loader is not None:
        dataloader_iter_t = iter(target_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    model.train()
    optimizer.zero_grad()

    for cur_it in range(total_it_each_epoch):
        lr_scheduler.step(accumulated_iter)
        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']
        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        # # source domain
        loss_s = torch.tensor(0.0).cuda()
        if source_loader is not None:        
            try:
                batch_s = next(dataloader_iter_s)
            except StopIteration:
                dataloader_iter_s = iter(source_loader)
                batch_s = next(dataloader_iter_s)
                print('new iters')
            ret_dict_s, tb_dict, disp_dict = model_func(model, batch_s)
            loss_s = ret_dict_s['loss'].mean()
            loss_s.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()
            # pts_all = batch_s['points'][:,1:4]     
            # visualize_utils.draw_scenes(pts_all, None, batch_s['gt_boxes'][0])


        loss_t = torch.tensor(0.0).cuda()
        if target_loader is not None:
            try:
                batch_t = next(dataloader_iter_t)
            except StopIteration:
                dataloader_iter_t = iter(target_loader)
                batch_t = next(dataloader_iter_t)
            ret_dict_t, _, _ = model_func(model, batch_t)
            #pts_all = batch_t['points'][:,1:4]            
            #visualize_utils.draw_scenes(pts_all, None, batch_t['gt_boxes'][0])
            loss_t = ret_dict_t['loss'].mean()      
            loss_t.backward()
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        accumulated_iter += 1
        disp_dict = {}
        tb_dict = {}        
        disp_dict.update({'loss_s': loss_s.item(), 'loss_t': loss_t.item(), 'lr': cur_lr})
        #disp_dict.update({'loss_s': loss_s.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss_s+loss_t, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter




def train_model_mine(model, optimizer, source_loader, target_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None, total_it_each_epoch=None, weights=[1.0,1.0]):
    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = total_it_each_epoch
        #total_it_each_epoch = len(target_loader)

        for cur_epoch in tbar:
            if source_sampler is not None:
                source_sampler.set_epoch(cur_epoch)
            
            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch_st(
                model, optimizer, source_loader, target_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch)

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:
                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )



def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
