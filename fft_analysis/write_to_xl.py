import xlsxwriter
import pickle


ods = ['ssd', 'yolo v3', 'faster rcnn', 'mask rcnn', 'retina-net']

for u in range(len(ods)):
    workbook = xlsxwriter.Workbook(f'{u}_models_signals.xlsx')

    ws = workbook.add_worksheet()


    for i in range(9):
        with open(f'models_results1/{i}_frames_models_pred', 'rb') as f:
            preds = pickle.load(f)
        pred = preds[u]
        for j in range(235):
            ws.write(j, i, pred[j])



    workbook.close()
