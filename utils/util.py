import torch
import numpy as np
import cv2

def DisplayHMI(input_path, labels_Seg, labels_Det, model_outputs):
    camera_image = cv2.imread(input_path[0])
    camera_image_normalized = camera_image.astype(np.float32) / 255.0
    camera_image_normalized = camera_image_normalized[:, :850, :]

    # Model outputs
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    pred_obj = pred_obj.squeeze(axis=0)
    pred_det = cv2.merge((pred_obj, pred_obj, pred_obj))
    pred_det = pred_det.astype(np.float32)
    pred_det = pred_det[:, :850, :]
    pred_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0]
    pred_seg = pred_seg.squeeze(axis=0)
    pred_seg = cv2.merge((pred_seg, pred_seg, pred_seg))
    pred_seg = pred_seg.astype(np.float32)

    labels_Det = labels_Det.detach().cpu().numpy().copy()[0]
    labels_Det = labels_Det.squeeze(axis=0)
    labels_Det = cv2.merge((labels_Det, labels_Det, labels_Det))
    labels_Det = labels_Det.astype(np.float32)
    labels_Det = labels_Det[:, :850, :]
    labels_Seg = labels_Seg.detach().cpu().numpy().copy()[0]
    labels_Seg = labels_Seg.astype(np.float32)
    labels_Seg = cv2.merge((labels_Seg, labels_Seg, labels_Seg))
    labels_Seg = labels_Seg.astype(np.float32)

    blend_GT = cv2.addWeighted(camera_image_normalized, 0.7, labels_Det, 0.3, 0)
    blend_GT = cv2.addWeighted(blend_GT, 0.7, labels_Seg, 0.3, 0)

    pred_GT = cv2.addWeighted(camera_image_normalized,0.7, pred_det, 0.3,0)
    pred_GT = cv2.addWeighted(pred_GT, 0.7, pred_seg, 0.3, 0)

    return np.hstack((blend_GT, pred_GT))