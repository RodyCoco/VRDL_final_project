import csv
import numpy as np

input1 = "Vit_large_patch16_384_bbox_1_bbox.csv"
iuput2 = "Vit_large_patch16_384_1.csv"
output = 'ensemble.csv'

with open(input1, 'r', newline='') as csvfile1:
    with open(iuput2, 'r', newline='') as csvfile2:
        rows1 = csv.reader(csvfile1)
        rows2 = csv.reader(csvfile2)
        rows1 = list(rows1)
        rows2 = list(rows2)
        with open(output, 'w', newline='') as csv2:
            writer = csv.writer(csv2)
            for idx, row in enumerate(rows1):
                if np.argmax(row[1:]) == 4:
                    row = rows2[idx]
                writer.writerow(row)
