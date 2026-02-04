import pickle
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

avgs = []
rngs = []
abv5 = []
abv6 = []
abv7 = []
abv8 = []
mins = []
maxs = []


def plot_compare():
    for i in range(4):
        with open(f'aug_{i}_videos_frames_pred', 'rb') as f:
            preds = pickle.load(f)

        yolo = preds[0]
        berkley = preds[1]
        aug = preds[2]

        # plt.plot(yolo, label='original', color='blue')
        plt.plot(berkley, label='berkley', color='orange')
        plt.title('Berkley Model - Confidence per Frame')
        # plt.plot(aug, label='fine tuned model', color='green')
        # plt.title('Fined Tuned Model - Confidence per Frame')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('confidence')
        plt.ylim(0, 1)
        plt.show()


def percentage_above_threshold(signal, threshole):
    above = np.asarray(signal) > threshole
    return sum(above) / len(signal)


def create_ensemble_signal(*signals):
    ensemble = []
    for i in range(len(signals[0])):
        ensemble.append(max([signal[i] for signal in signals]))
    return ensemble


def print_stats(signal, ws, i):
    avarage = np.average(signal)
    avgs.append(avarage)
    absolute_range = np.max(signal) - np.min(signal)
    rngs.append(absolute_range)
    above_5 = percentage_above_threshold(signal, 0.5)
    abv5.append(above_5)
    above_6 = percentage_above_threshold(signal, 0.6)
    abv6.append(above_6)
    above_7 = percentage_above_threshold(signal, 0.7)
    abv7.append(above_7)
    above_8 = percentage_above_threshold(signal, 0.8)
    abv8.append(above_8)
    minimum_value = np.min(signal)
    mins.append(minimum_value)
    maximum_value = np.max(signal)
    maxs.append(maximum_value)

    print('average: ', avarage)
    print('absolute range: ', absolute_range)
    print('above 0.5', above_5)
    print('above 0.6: ', above_6)
    print('above 0.7: ', above_7)
    print('above 0.8: ', above_8)
    print('minimum value: ', minimum_value)
    print('maximum vakue: ', maximum_value)

    ws.write(i + 1, 0, avarage)
    ws.write(i + 1, 1, absolute_range)
    ws.write(i + 1, 2, above_5)
    ws.write(i + 1, 3, above_6)
    ws.write(i + 1, 4, above_7)
    ws.write(i + 1, 5, above_8)
    ws.write(i + 1, 6, minimum_value)
    ws.write(i + 1, 7, maximum_value)


def sort_by(e):
    return e[1]


denoising_model = 3
eval_model = 3
eval_model_dict = {0: 'yolo', 1: 'berkley', 2: 'aug', 3: 'ensemble'}


def write_titles(ws):
    print('********* averages:')
    print('average: ', np.average(avgs))
    print('absolute range: ', np.average(rngs))
    print('above 0.5: ', np.average(abv5))
    print('above 0.6: ', np.average(abv6))
    print('above 0.7: ', np.average(abv7))
    print('above 0.8: ', np.average(abv8))
    print('minimum value: ', np.average(mins))
    print('maximum value: ', np.average(maxs))

    ws.write(0, 0, 'average')
    ws.write(0, 1, 'absolute range')
    ws.write(0, 2, 'above 0.5')
    ws.write(0, 3, 'above 0.6')
    ws.write(0, 4, 'above 0.7')
    ws.write(0, 5, 'above 0.8')
    ws.write(0, 6, 'minimum value')
    ws.write(0, 7, 'maximum value')
    ws.write(3, 9, 'model average')
    ws.write(3, 10, 'average')
    ws.write(3, 11, 'absolute range')
    ws.write(3, 12, 'above 0.5')
    ws.write(3, 13, 'above 0.6')
    ws.write(3, 14, 'above 0.7')
    ws.write(3, 15, 'above 0.8')
    ws.write(3, 16, 'minimum value')
    ws.write(3, 17, 'maximum value')
    ws.write(3, 18, 'consistent improvement')
    ws.write(3, 19, 'consistent failure')
    ws.write(3, 20, 'inconsistent result')

    ws.write(4, 10, np.average(avgs))
    ws.write(4, 11, np.average(rngs))
    ws.write(4, 12, np.average(abv5))
    ws.write(4, 13, np.average(abv6))
    ws.write(4, 14, np.average(abv7))
    ws.write(4, 15, np.average(abv8))
    ws.write(4, 16, np.average(mins))
    ws.write(4, 17, np.average(maxs))


def consist_results(ws, signal, baseline):
    num_of_videos = 243

    amount_lower = 0
    amount_higher = 0
    amount_grey = 0
    if np.min(signal) > np.max(baseline):
        amount_higher += 1
    elif np.min(baseline) > np.max(signal):
        amount_lower += 1
    else:
        amount_grey += 1
    print('consistent improvement', amount_higher / num_of_videos)
    print('consistent failure', amount_lower / num_of_videos)
    print('inconsistent result', amount_grey / num_of_videos)

    ws.write(4, 18, amount_higher / num_of_videos)
    ws.write(4, 19, amount_lower / num_of_videos)
    ws.write(4, 20, amount_grey / num_of_videos)


def statistics():
    workbook = xlsxwriter.Workbook(f'new_new_results/_____.xlsx')

    ws = workbook.add_worksheet()
    s = []

    num_of_videos = 242
    for i in range(num_of_videos):
        with open(f'./denoising preds/siren{denoising_model}/aug_{i}_siren{denoising_model}_results_pred', 'rb') as f:
            preds = pickle.load(f)
        with open(f'original_preds/aug_{i}_videos_eval_pred', 'rb') as f:
            preds_orig = pickle.load(f)
        with open(f'./cyclegan_model_preds_new/aug_{i}_videos_eval_pred', 'rb') as f:
            cyclegan_aug_preds = pickle.load(f)
        with open(f'./cyclegan_size/aug_{i}_videos_eval_pred', 'rb') as f:
            cyclegan_size_preds = pickle.load(f)
        with open(f'./super/super_aug_{i}_videos_eval_pred', 'rb') as f:
            cyclegan_super = pickle.load(f)
        with open(f'./super_denoising/super_denoise_aug_{i}_denoise_pred', 'rb') as f:
            super_denoising = pickle.load(f)
        with open(f'./denoise_results/denoise_{i}_denoise_pred', 'rb') as f:
            denoise_new = pickle.load(f)
        with open(f'./h_h_results/h_h{i}_videos_eval_pred', 'rb') as f:
            h_h_preds = pickle.load(f)
        with open(f'./denoise3_results_new_new/denoise_fake_3_{i}_videos_eval_pred', 'rb') as f:
            denoise3_preds = pickle.load(f)

        denoising_yolo_new = denoise_new[0]
        denoising_berkley_new = denoise_new[1]
        denoising_aug_new = denoise_new[2]

        denoising_yolo = preds[0]
        denoising_berkley = preds[1]
        denoising_aug = preds[2]

        yolo = preds_orig[0]
        berkley = preds_orig[1]
        aug = preds_orig[2]

        big = cyclegan_size_preds[0]
        small = cyclegan_size_preds[1]

        ensemble = create_ensemble_signal(berkley, preds[2], cyclegan_aug_preds, preds_orig[2], big, small)
        super = cyclegan_super
        super_denoising = super_denoising
        s.append((i, np.average(cyclegan_aug_preds)))


        h_h_50 = h_h_preds[0]
        h_h_100 = h_h_preds[1]
        h_h_150 = h_h_preds[2]
        h_h_200 = h_h_preds[3]

        denoise3_yolo = denoise3_preds[0]
        denoise3_berkley = denoise3_preds[1]
        denoise3_aug = denoise3_preds[2]
        print(f'\nVideo number {i}')
        # print_stats(preds[eval_model] if eval_model < 3 else ensemble, ws, i)
        print_stats(denoising_yolo_new, ws, i)

    write_titles(ws)

    workbook.close()

    s.sort(key=sort_by)


statistics()
