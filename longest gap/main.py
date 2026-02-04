import pickle
import numpy as np
from matplotlib import pyplot as plt

fps = 30
miliseconds_per_frame = 1000 / 30
ods = ['ssd', 'yolo3', 'frcnn', 'mrcnn', 'retina-net']
threshold = 0.7

def longest_gap(signal, threshold):
    above_threshold = True
    current_gap = 0
    longest_gap = 0
    decrease_start = None
    max_point = (-1, -1)

    for i, amplitude in enumerate(signal):
        if amplitude >= threshold:
            if not above_threshold:
                above_threshold = True
                current_gap = 0
                if decrease_start is not None:
                    # Plot a dotted line at the highest point
                    plt.plot(max_point[0], max_point[1], 'ro', linestyle='dotted')
                    max_point = (-1, -1)
                    decrease_start = None
            else:
                longest_gap = max(longest_gap, current_gap)
                current_gap = 0
        else:
            current_gap += 1
            if above_threshold and (decrease_start is None or amplitude > max_point[1]):
                decrease_start = i
                max_point = (i, amplitude)

    return longest_gap

with open('car_4_preds', 'rb') as f:
    signals = pickle.load(f)

for object_detector, _ in enumerate(ods):
    plt.plot(signals[object_detector])
    plt.ylim(0, 1.05)
    plt.show()
    gaps = []
    thresholds = []
    max_gap = 0
    max_threshold = 0
    last_gap = 0
    dotted_thres = []
    dotted_gap = []
    for i in range(0, 100):
        threshold = i * 0.01
        res = longest_gap(signals[object_detector], threshold)
        gap_in_seconds = res / 30
        if res < last_gap:
            dotted_thres.append(threshold)
            dotted_gap.append(last_gap / 30)
        else:
            last_gap = res
        thresholds.append(threshold)
        gaps.append(gap_in_seconds)
        if res > max_gap:
            max_gap = res
            max_threshold = threshold

        print(f"{ods[object_detector]} - The gap longest gap between the Threshold {threshold} is {res}")
    print(f'Max gap is {max_gap} with threshold of: {max_threshold}')

    plt.title(f'{ods[object_detector]} - Gap duration of the Longest Gap Across Thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Time (Sec)')
    plt.plot(thresholds, gaps)
    plt.plot(dotted_thres, dotted_gap, linestyle='dotted')
    plt.show()
