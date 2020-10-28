import logreg
import xlrd
from sklearn import linear_model
import numpy as np

data = xlrd.open_workbook('melon.xlsx')
table = data.sheet_by_index(0)
dataFile = []
for rowNum in range(table.nrows):
    dataFile.append(table.row_values(rowNum))

A = logreg.logreg(dataFile, 0.1)
print(A)

data_len = len(dataFile)
par_len = len(dataFile[0])
y = []
for i in range(data_len):
    y.append(dataFile[i][par_len - 1])
    dataFile[i].pop()

X = np.array(dataFile)
y = np.array(y)

classifier = linear_model.LogisticRegression(solver='liblinear', C=100)
classifier.fit(X, y)