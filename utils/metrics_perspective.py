from sklearn.metrics import average_precision_score, recall_score, precision_score
from sklearn.metrics import jaccard_score, f1_score

threshold = 0.5

def calculate_mAP_mAR(label_map, outputs):
    # Reshape the outputs and labels to be 1D arrays
    outputs_flat = outputs.view(-1).cpu().numpy()
    # label_map_flat = label_map.view(-1).cpu().numpy()
    label_map_flat = label_map.view(-1).cpu().numpy()

    # Calculate average precision
    mAP = average_precision_score(label_map_flat, outputs_flat)

    # Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
    outputs_binary = (outputs_flat > threshold).astype(int)

    # Calculate recall
    mAR = recall_score(label_map_flat, outputs_binary, zero_division=1)

    return mAP, mAR

def calculate_mean_iou(seg_labels, seg_predictions):
    # Flatten the tensors along all dimensions except the batch dimension
    seg_labels_flat = seg_labels.view(seg_labels.size(0), -1).cpu().numpy()
    seg_predictions_flat = (seg_predictions > threshold).view(seg_predictions.size(0), -1).cpu().numpy()

    mIoU = 0.0

    for i in range(seg_labels.size(0)):
        # Calculate IoU for each sample in the batch
        iou_sample = jaccard_score(seg_labels_flat[i], seg_predictions_flat[i])
        mIoU += iou_sample

    # Calculate mean IoU
    mIoU /= seg_labels.size(0)

    return mIoU

def calculate_mAP_mAR_eval(label_map, outputs):
    # Reshape the outputs and labels to be 1D arrays
    outputs_flat = outputs.flatten()
    label_map_flat = label_map.flatten()
    # outputs.flatten().cpu().numpy()
    # Calculate average precision
    mAP = average_precision_score(label_map_flat, outputs_flat)

    # Convert predictions to binary (0 or 1) based on a threshold (e.g., 0.5)
    outputs_binary = (outputs_flat > threshold).astype(int)

    # Calculate recall
    mAR = recall_score(label_map_flat, outputs_binary, average='macro', zero_division=1)

    precision = precision_score(label_map_flat, outputs_binary, average='macro', zero_division=1)

    # F1_score = (mAP * mAR) / ((mAP + mAR) / 2)
    F1_score = (2 * precision * mAR) / (precision + mAR)

    return mAP, mAR, F1_score

def calculate_mean_iou_eval(seg_labels, seg_predictions):
    # Flatten the tensors along all dimensions except the batch dimension
    labels_flat = seg_labels.reshape(seg_labels.shape[0], -1)
    seg_predictions = seg_predictions.squeeze(1)
    seg_predictions = seg_predictions.reshape(seg_predictions.shape[0], -1)
    predictions_binary = (seg_predictions > threshold).astype(int) #.reshape(seg_predictions.size(0), -1)

    mIoU = 0.0

    for i in range(seg_labels.shape[0]):
        # Calculate IoU for each sample in the batch
        iou_sample = jaccard_score(labels_flat[i], predictions_binary[i])
        mIoU += iou_sample

    # Calculate mean IoU
    mIoU /= seg_labels.shape[0]

    return mIoU

