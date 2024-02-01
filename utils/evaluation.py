import torch
import numpy as np
from .metrics_bev import Metrics, Metrics_seg, Metrics_det, GetFullMetrics
from .metrics_perspective import *
import pkbar

def run_evaluation(net, loader, device,config,encoder, detection_loss=None, segmentation_loss=None, losses_params=None,mode_params=None):

    if config['model']['view_birdseye'] == 'True':
        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
            metrics = Metrics()
            metrics.reset()
        if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
            metrics_seg = Metrics_seg()
            metrics_seg.reset()
        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
            metrics_det = Metrics_det()
            metrics_det.reset()

    net.eval()
    classif_loss = torch.tensor(0, dtype=torch.float64)
    reg_loss = torch.tensor(0, dtype=torch.float64)
    loss_seg = torch.tensor(0, dtype=torch.float64)
    running_loss = 0.0

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):
        if (config['architecture']['perspective']['only_camera'] == 'True' or
            config['architecture']['perspective']['early_fusion'] == 'True' or
            config['architecture']['bev']['only_radar'] == 'True'):
            inputs1 = data[0].to(device).float() # fusion is done on dataset.py
            seg_map_label = data[1].to(device).double()
            det_label = data[2].to(device).float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs1)

        if (config['architecture']['perspective']['after_decoder_fusion'] == 'True' or
            config['architecture']['perspective']['x4_fusion'] == 'True' or
            config['architecture']['bev']['early_fusion'] == 'True'or
            config['architecture']['bev']['after_decoder_fusion'] == 'True' or
            config['architecture']['bev']['x4_fusion'] == 'True'):
            inputs1 = data[0].to(device).float()
            inputs2 = data[1].to(device).float()
            seg_map_label = data[2].to(device).double()
            det_label = data[3].to(device).float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs2, inputs1)

        if (config['model']['view_birdseye'] == 'True' and detection_loss != None and segmentation_loss != None):
            if config['model']['DetectionHead'] == 'True':
                classif_loss, reg_loss = detection_loss(outputs['Detection'], det_label, losses_params,mode_params)
                classif_loss *= losses_params['weight'][0]
                reg_loss *= losses_params['weight'][1]
            if config['model']['SegmentationHead'] == 'True':
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = segmentation_loss(prediction, label)
                loss_seg *= inputs1.size(0)
                loss_seg *= losses_params['weight'][2]

            loss = classif_loss + reg_loss + loss_seg
            running_loss += loss.item() * inputs1.size(0)

            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                out_obj = outputs['Detection'].detach().cpu().numpy().copy()
                if config['architecture']['bev']['only_radar']=='False':
                    labels = data[4]
                else:
                    labels = data[3]
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                for pred_obj, pred_map, true_obj, true_map in zip(out_obj, out_seg, labels, label_freespace):
                    metrics.update(pred_map[0], true_map, np.asarray(encoder.decode(pred_obj, 0.05)), true_obj,
                                   threshold=0.2, range_min=5, range_max=100)
            if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                for pred_map, true_map in zip(out_seg, label_freespace):
                    metrics_seg.update(pred_map[0], true_map, threshold=0.2,
                                   range_min=5, range_max=100)
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
                out_obj = outputs['Detection'].detach().cpu().numpy().copy()
                if config['architecture']['bev']['only_radar']=='False':
                    labels = data[4]
                else:
                    labels = data[3]
                for pred_obj, true_obj in zip(out_obj, labels):
                    metrics_det.update(np.asarray(encoder.decode(pred_obj, 0.05)), true_obj,
                                   threshold=0.2, range_min=5, range_max=100)

        else: #perspective view
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                classif_loss, reg_loss = detection_loss(outputs['Detection'], det_label, losses_params,mode_params)
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = segmentation_loss(prediction, label)
                loss_seg *= inputs1.size(0)
                classif_loss *= losses_params['weight'][0]
                loss_seg *= losses_params['weight'][2]
                loss = classif_loss + reg_loss +loss_seg
                running_loss += loss.item() * inputs1.size(0)
            if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()
                loss_seg = segmentation_loss(prediction, label)
                loss_seg *= inputs1.size(0)
                loss_seg *= losses_params['weight'][2]
                loss = classif_loss + reg_loss + loss_seg
                running_loss += loss.item() * inputs1.size(0)
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
                classif_loss, reg_loss = detection_loss(outputs['Detection'], det_label, losses_params, mode_params)
                classif_loss *= losses_params['weight'][0]
                loss = classif_loss + reg_loss + loss_seg
                running_loss += loss.item() * inputs1.size(0)
        kbar.update(i)

    if (config['model']['view_birdseye'] == 'True'):
        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
            mAP, mAR, mIoU = metrics.GetMetrics()
            return {'loss': running_loss, 'mAP': mAP, 'mAR': mAR, 'mIoU': mIoU}
        if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
            mIoU = metrics_seg.GetMetrics()
            return {'loss': running_loss, 'mIoU': mIoU}
        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
            mAP, mAR = metrics_det.GetMetrics()
            return {'loss': running_loss, 'mAP': mAP, 'mAR': mAR}
    else:
        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
            mAP, mAR = calculate_mAP_mAR(det_label, outputs['Detection'])
            mIoU = calculate_mean_iou(seg_map_label, outputs['Segmentation'])
            return {'loss': running_loss, 'mAP': mAP, 'mAR': mAR, 'mIoU': mIoU}
        if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
            mIoU = calculate_mean_iou(seg_map_label, outputs['Segmentation'])
            return {'loss': running_loss, 'mIoU': mIoU}
        if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'False':
            mAP, mAR = calculate_mAP_mAR(det_label, outputs['Detection'])
            return {'loss': running_loss, 'mAP': mAP, 'mAR': mAR}


def run_FullEvaluation(net,loader,device,config,encoder):

    net.eval()
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)
    print('Generating Predictions...')

    total_mAP = 0.0
    total_mAR = 0.0
    total_F1_score = 0.0
    total_mIoU = 0.0
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}

    for i, data in enumerate(loader):
        # batch_start_memory = torch.cuda.memory_allocated() / (1024 ** 2)

        if (config['architecture']['perspective']['only_camera'] == 'True' or
            config['architecture']['perspective']['early_fusion'] == 'True' or
            config['architecture']['bev']['only_radar'] == 'True'):
            inputs1 = data[0].to(device).float()
            seg_map_label = data[1].to(device).double()
            det_label = data[2].to(device).float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs1)

        if (config['architecture']['perspective']['after_decoder_fusion'] == 'True' or
            config['architecture']['perspective']['x4_fusion'] == 'True' or
            config['architecture']['bev']['early_fusion'] == 'True'or
            config['architecture']['bev']['after_decoder_fusion'] == 'True' or
            config['architecture']['bev']['x4_fusion'] == 'True'):
            inputs1 = data[0].to(device).float()
            inputs2 = data[1].to(device).float()
            seg_map_label = data[2].to(device).double()
            det_label = data[3].to(device).float()
            with torch.set_grad_enabled(False):
                outputs = net(inputs2, inputs1)

        if config['model']['view_birdseye'] == 'True':
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                out_obj = outputs['Detection'].detach().cpu().numpy().copy()
                if config['architecture']['bev']['only_radar']=='False':
                    labels = data[4]
                else:
                    labels = data[3]
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                for pred_obj, pred_map, true_obj, true_map in zip(out_obj, out_seg, labels, label_freespace):
                    predictions['prediction']['objects'].append(np.asarray(encoder.decode(pred_obj, 0.05)))
                    predictions['label']['objects'].append(true_obj)
                    predictions['prediction']['freespace'].append(pred_map[0])
                    predictions['label']['freespace'].append(true_map)
                # batch_end_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                # print(f"Batch {i + 1} - GPU Memory Usage: {batch_end_memory:.2f} MiB (Increase: {batch_end_memory - batch_start_memory:.2f} MiB)")
                kbar.update(i)
            if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                for pred_map, true_map in zip(out_seg, label_freespace):
                    predictions['prediction']['freespace'].append(pred_map[0])
                    predictions['label']['freespace'].append(true_map)
                # batch_end_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                # print(f"Batch {i + 1} - GPU Memory Usage: {batch_end_memory:.2f} MiB (Increase: {batch_end_memory - batch_start_memory:.2f} MiB)")
                kbar.update(i)

        else: #perspective view
            if config['model']['DetectionHead'] == 'True' and config['model']['SegmentationHead'] == 'True':
                out_obj = outputs['Detection'].detach().cpu().numpy().copy()
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                det_label = det_label.detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                mAP, mAR, F1_score = calculate_mAP_mAR_eval(det_label, out_obj)
                mIoU = calculate_mean_iou_eval(label_freespace, out_seg)
                total_mAP += mAP
                total_mAR += mAR
                total_F1_score += F1_score
                total_mIoU += mIoU
                # batch_end_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                # print(f"Batch {i + 1} - GPU Memory Usage: {batch_end_memory:.2f} MiB (Increase: {batch_end_memory - batch_start_memory:.2f} MiB)")
                kbar.update(i)
            if config['model']['DetectionHead'] == 'False' and config['model']['SegmentationHead'] == 'True':
                out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
                label_freespace = seg_map_label.detach().cpu().numpy().copy()
                mIoU = calculate_mean_iou_eval(label_freespace, out_seg)
                total_mIoU += mIoU
                # batch_end_memory = torch.cuda.memory_allocated() / (1024 ** 2)
                # print(f"Batch {i + 1} - GPU Memory Usage: {batch_end_memory:.2f} MiB (Increase: {batch_end_memory - batch_start_memory:.2f} MiB)")
                kbar.update(i)

    if config['model']['view_birdseye'] == 'True':
        if config['model']['DetectionHead'] == 'True':
            GetFullMetrics(predictions['prediction']['objects'], predictions['label']['objects'], range_min=5,
                           range_max=100, IOU_threshold=0.5)
        if (config['model']['SegmentationHead'] == 'True'):
            mIoU = []
            for i in range(len(predictions['prediction']['freespace'])):
                # 0 to 124 means 0 to 50m
                pred = predictions['prediction']['freespace'][i][:124].reshape(-1) >= 0.5
                label = predictions['label']['freespace'][i][:124].reshape(-1)

                intersection = np.abs(pred * label).sum()
                union = np.sum(label) + np.sum(pred) - intersection
                iou = intersection / union
                mIoU.append(iou)

            mIoU = np.asarray(mIoU).mean()
            print('------- Freespace Scores ------------')
            print('  mIoU', mIoU * 100, '%')

    else:
        if config['model']['DetectionHead'] == 'True':
            average_mAP = total_mAP / len(loader)
            average_mAR = total_mAR / len(loader)
            average_f1 = total_F1_score / len(loader)
            print('------- Detection Scores ------------')
            print("Average mAP:", average_mAP)
            print("Average mAR:", average_mAR)
            print('  F1 score:', average_f1)
        if (config['model']['SegmentationHead'] == 'True'):
            average_miou = total_mIoU / len(loader)
            print('------- Freespace Scores ------------')
            print("Average mIoU:", average_miou)
