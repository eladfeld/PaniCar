import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import os
import time
def convert_result_file():
    with open('yolo_berkley', 'rb') as f:
        yolo_berkley = pickle.load(f)
    with open('yolo_aug', 'rb') as f:
        yolo_aug = pickle.load(f)
    with open('yolo_cyclegan', 'rb') as f:
        yolo_cyclegan = pickle.load(f)

    results = []
    for i in range(len(yolo_berkley)):
        results.append((yolo_berkley[i][0], (yolo_berkley[i][1], yolo_aug[i][1], yolo_cyclegan[i][1])))
    return results


def ensemble_predictions(models_predictions, weights):
    num_classes = len(models_predictions[0])
    ensemble_result = []

    for class_idx in range(num_classes):
        class_predictions = []
        for model_preds, weight in zip(models_predictions, weights):
            class_predictions.extend(model_preds[class_idx])

        # Sort predictions by score in descending order
        class_predictions.sort(key=lambda x: x[4], reverse=True)

        # Calculate weighted average score
        total_score = 0.0
        total_weight = 0.0
        for bbox in class_predictions:
            total_score += bbox[4] * weight
            total_weight += weight

        weighted_avg_score = total_score / total_weight if total_weight > 0 else 0.0
        ensemble_result.append(weighted_avg_score)

    return ensemble_result


def ensemble_object_detection(preds, threshold=0.5, iou_threshold=0.5):
    """
    Creates an ensemble object detection model using a list of individual object detection models.

    Args:
        models (list): A list of object detection models.
        image (array-like): The input image for object detection.
        threshold (float): Confidence threshold for object detection.
        iou_threshold (float): Intersection over Union (IoU) threshold for non-maximum suppression.

    Returns:
        List of dictionaries, where each dictionary represents a detected object with keys 'class', 'score',
        'bbox' (bounding box coordinates).
    """

    num_of_classes = len(preds[0])
    ensemble_result = [[] for i in range(num_of_classes)]
    for pred in preds:
        for clas in range(num_of_classes):
            ensemble_result[clas].extend(pred[clas])
    nms_predictions = non_max_suppression(ensemble_result, num_of_classes, iou_threshold)

    # Filter predictions based on confidence threshold
    for i, clas in enumerate(nms_predictions):
        nms_predictions[i] = [pred for pred in clas if pred[4] >= threshold]

    return nms_predictions


def plot_image_with_bboxes(image_path, detection_results):
    # Load the image
    image = Image.open(image_path)

    # Create a figure and axis
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Define a list of colors for bounding box rectangles
    bbox_colors = ['r', 'g', 'b', 'y', 'm', 'c', 'orange', 'pink', 'purple', 'brown']

    # Iterate over each class's detection results
    for class_idx, class_data in enumerate(detection_results):
        for bbox_data in class_data:
            x_min, y_min, x_max, y_max, score = bbox_data
            width = x_max - x_min
            height = y_max - y_min

            bbox = patches.Rectangle((x_min, y_min), width, height, linewidth=1,
                                     edgecolor=bbox_colors[class_idx], facecolor='none')

            # Add the bounding box to the plot
            ax.add_patch(bbox)

            # Add the score as a label near the bounding box
            # ax.text(x_min, y_min - 10, f"{score:.2f}", color=bbox_colors[class_idx],
            #         fontsize=10, ha="center", bbox=dict(facecolor='white', alpha=0.7))

    # Show the plot
    plt.axis('off')
    plt.show()

def non_max_suppression(predictions_by_class, num_of_classes, iou_threshold):
    """
    Applies non-maximum suppression to a list of object detection predictions.

    Args:
        predictions (list): List of dictionaries representing object detection predictions.
        iou_threshold (float): Intersection over Union (IoU) threshold.

    Returns:
        List of dictionaries after applying non-maximum suppression.
    """

    nms_predictions = [[] for _ in range(num_of_classes)]

    # Apply NMS for each class separately
    for i, class_preds in enumerate(predictions_by_class):
        class_preds.sort(key=lambda x: x[4], reverse=True)
        selected_preds = []

        for pred in class_preds:
            if all(calc_iou(pred, selected_pred) < iou_threshold for selected_pred in selected_preds):
                selected_preds.append(pred)

        nms_predictions[i].extend(selected_preds)

    return nms_predictions

def calc_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (dict): Dictionary representing the first bounding box with keys 'bbox'.
        box2 (dict): Dictionary representing the second bounding box with keys 'bbox'.

    Returns:
        IoU value between the two bounding boxes.
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate area of intersection and union
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area
    return iou

all_results = convert_result_file()

base_path = "C:\\Users\\Administrator\\Desktop\\IoU\\val"
x = os.listdir(base_path)
times = []
for image_name, preds in all_results:
    start_time = time.time()
    ensemble_result = ensemble_object_detection(preds)
    times.append(time.time() - start_time)
    # plot_image_with_bboxes(os.path.join(base_path, image_name), ensemble_result)

print(np.mean(times))




