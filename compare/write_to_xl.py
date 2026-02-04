import xlsxwriter
import pickle


with open('0__pred', 'rb') as f:
    preds = pickle.load(f)

workbook = xlsxwriter.Workbook('xl.xlsx')

ws = workbook.add_worksheet()
ods = ['ssd', 'yolo v3', 'faster rcnn', 'mask rcnn', 'retina-net']


pred = preds[0]
for i in range(len(pred)):
    ws.write(i, 0, preds[0][i])
    ws.write(i, 1, preds[1][i])
    ws.write(i, 2, preds[2][i])


workbook.close()
