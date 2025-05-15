# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os, pdb
import math
import numpy as np
from typing import Iterable, Optional

import torch
import torch.distributed as dist
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
from utils import adjust_learning_rate
from scipy.special import softmax
from sklearn.metrics import (
    average_precision_score, 
    accuracy_score
)

# from data.vis_cam import visual





def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    update_freq = args.update_freq
    use_amp = args.use_amp
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq=500, header=header)):

        samples = batch[0]
        targets = batch[-1]
        
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % update_freq == 0:
            adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        for ind, _ in enumerate(samples):
            samples[ind] = samples[ind].to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                # loss = criterion(output, targets)
        else: # full precision
            output = model(samples)[0]
            ##!这里的对比损失还没修改.可以用度量损失来拉近关系和对比结构
            #*对比损失是否都放在一个球面上还没想好
            
            alph = 0.2
            loss0 = criterion['class'](output, targets)
            # loss1 = criterion[0](output[1], targets)
            # loss2 = criterion[0](output[2], targets)
            #下面这个就是一维数据.所以直接fft就可以了
            # print("output[0]:",output[0].shape,"targets:",targets.shape)
            #?这个不能用,看看是否能改成其他部分来使用?
            loss_feq = (torch.fft.rfft(output, dim=1) - torch.fft.rfft(targets.unsqueeze(1), dim=1)).abs().mean() #因为不希望对差异敏感,所以用mae,可以后期改成smooth之类的
            # # loss1 = criterion[1](output[1], -output[2]).mean() + criterion[1](output[3], -output[4]).mean()
            # loss2 = criterion[1](output[1], output[3]).mean() + criterion[1](output[2], output[4]).mean()#*这都是简单的使用方式,不一定好用.要多试一试.
            
            loss = (1 - alph) * loss0 + alph * loss_feq 
            #  + loss1 + loss2
            # loss = criterion(output, targets)
        loss_value = loss.item()

        

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        
        torch.cuda.synchronize()
        #一个batch的准确率
        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()#二分类任务的output.max获得的是最大值和索引.
        else:
            class_acc = None

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)
        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, val=None, use_amp=False):
    # print("开始测试")
    criterion = torch.nn.CrossEntropyLoss()#?这里为啥不用smooth

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    a = []
    b = []
    for index, batch in enumerate(metric_logger.log_every(data_loader, 500, header)):
        # print("数据:",batch[0].shape,batch[0])
        images = batch[0]
        target = batch[-1]
       
        # a.extend(images[3].numpy().tolist())
        # b.extend(target.numpy().tolist())

        for ind, _ in enumerate(images):
            images[ind] = images[ind].to(device, non_blocking=True)
        # images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # if index ==1:
        #     visual(model,images,images[-1], target, use_cuda=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                if isinstance(output, dict):
                    output = output['logits']
                loss = criterion(output, target)
        else:
            # output,cat,bran1,bran2 = model(images) #[bs, num_cls]
            output,cat,bran1,bran2,bran3 = model(images)
            if isinstance(output, dict):
                output = output['logits']

            #*loss,好像只是简单记录一下而已.反正没用,所以不需要完全一致??
            alph = 0.2
            
            loss0 = criterion(output, target)
            # loss1 = criterion[0](output[1], target)
            # loss2 = criterion[0](output[2], target)
            loss_feq = (torch.fft.rfft(output, dim=1) - torch.fft.rfft(target.unsqueeze(1), dim=1)).abs().mean() #因为不希望对差异敏感,所以用mae,可以后期改成smooth之类的
            # # loss1 = criterion[1](output[1], -output[2]).mean() + criterion[1](output[3], -output[4]).mean()
            # loss2 = criterion[1](output[1], output[3]).mean() + criterion[1](output[2], output[4]).mean()#*这都是简单的使用方式,不一定好用.要多试一试.
            
            loss = (1 - alph) * loss0 + alph * loss_feq 

            
            # loss = criterion(output[0], target)
 
        if index == 0:
            predictions = output
            labels = target
            # catall = cat
            # branall = bran1 
            # bran2all =bran2
            # bran3all =bran3
            
        else:
            predictions = torch.cat((predictions, output), 0)
            # catall = torch.cat((catall, cat), 0)
            # branall = torch.cat((branall, bran1), 0)
            # bran2all = torch.cat((bran2all, bran2), 0)
            # bran3all = torch.cat((bran3all, bran3), 0)
            labels = torch.cat((labels, target), 0)

        torch.cuda.synchronize()
        #*分别计算不同类别
        # acc1_per_class = [acc1,acc1]
        # for cls in [0, 1]:
        #     cls_mask = (target == cls)
        #     if cls_mask.sum() > 0:
        #         acc1_cls, _ = [acc / 100 for acc in accuracy(output[cls_mask], target[cls_mask], topk=(1, 2))]
        #         acc1_per_class[cls] = acc1_cls.item()
        #     else:
        #         acc1_per_class[cls] = None
        #*这里修改,不如只取正或者负来训练
        acc1, _ = [acc / 100 for acc in accuracy(output, target, topk=(1, 2))]

        # acc1_per_class = [acc1,acc1]
# 
        batch_size = images[0].shape[0]
        metric_logger.update(loss=loss.item())
        # metric_logger.meters['acc1_label0'].update( acc1_per_class[0].item(), n=batch_size)
        # metric_logger.meters['acc1_label1'].update(acc1_per_class[1].item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    #循环结束#?为啥这个内存没爆炸哦,但是之前就爆炸了?果然还是放在显存好?
    # c = catall.detach().cpu().numpy()
    # d = branall.detach().cpu().numpy()
    # e = bran2all.detach().cpu().numpy()
    # f = bran3all.detach().cpu().numpy()
    # b = labels.detach().cpu().numpy() 
    # np.save('results/datasave/1.npy', (b,c,d,e))
    # np.save('visual/datasave/b1.npy', b)
    # np.save('visual/datasave/c1.npy', c)
    # np.save('visual/datasave/d1.npy', d)
    # np.save('visual/datasave/e1.npy', e)
    # np.save('visual/datasave/f1.npy', f)
    # print("数据保存完毕")
    # np.save('results/datasave/1.npy', a)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.2%} loss {losses.global_avg:.4f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    # print('* Acc@label0 {label0.global_avg:.2%}  Acc@label1 {label1.global_avg:.2%}  loss {losses.global_avg:.4f}'
    #       .format(label0=metric_logger.acc1_label0, label1=metric_logger.acc1_label1, losses=metric_logger.loss))

    output_ddp = [torch.zeros_like(predictions) for _ in range(utils.get_world_size())]
    dist.all_gather(output_ddp, predictions)
    labels_ddp = [torch.zeros_like(labels) for _ in range(utils.get_world_size())]
    dist.all_gather(labels_ddp, labels)

    output_all = torch.concat(output_ddp, dim=0)
    labels_all = torch.concat(labels_ddp, dim=0)

    y_pred = softmax(output_all.detach().cpu().numpy(), axis=1)[:, 1]
    y_true = labels_all.detach().cpu().numpy()
    y_true = y_true.astype(int)
  
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, ap


@torch.no_grad()
def visual_evaluate(data_loader, model, device, val=None, use_amp=False):
    # print("开始测试")
    criterion = torch.nn.CrossEntropyLoss()#?这里为啥不用smooth

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    a = []
    b = []
    for index, batch in enumerate(metric_logger.log_every(data_loader, 500, header)):
        # print("数据:",batch[0].shape,batch[0])
        images = batch[0]
        target = batch[-1]
        for ind, _ in enumerate(images):
            images[ind] = images[ind].to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # output,cat,bran1,bran2 = model(images) #[bs, num_cls]
        output,cat,bran1,bran2,bran3 = model(images)
        if index == 0:
            predictions = output
            labels = target
            # catall = cat
            # branall = bran1 
            # bran2all =bran2
            bran3all =bran3
        else:
            predictions = torch.cat((predictions, output), 0)
            # catall = torch.cat((catall, cat), 0)
            # branall = torch.cat((branall, bran1), 0)
            # bran2all = torch.cat((bran2all, bran2), 0)
            bran3all = torch.cat((bran3all, bran3), 0)
            labels = torch.cat((labels, target), 0)

        torch.cuda.synchronize()
    
    #循环结束#?为啥这个内存没爆炸哦,但是之前就爆炸了?果然还是放在显存好?
    # c = catall.detach().cpu().numpy()
    # d = branall.detach().cpu().numpy()
    # e = bran2all.detach().cpu().numpy()
    f = bran3all.detach().cpu().numpy()
    # b = labels.detach().cpu().numpy() 
    # np.save('results/datasave/1.npy', (b,c,d,e))
    # np.save('visual/datasave/b1.npy', b)
    # np.save('visual/datasave/c1.npy', c)
    # np.save('visual/datasave/d1.npy', d)
    # np.save('visual/datasave/e1.npy', e)
    np.save('visual/datasave/f1.npy', f)
    print("数据保存完毕")

# def cam():