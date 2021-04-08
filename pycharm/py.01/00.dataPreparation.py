import pandas as pd
import numpy as np

def ConvertStrToNumColumn(col):
    # print(col)
    if( type(col[0]) == str):
        strDic = {}
        xCategory = 0
        for i, row in enumerate(col):
            if row in strDic.keys():
                col[i] = strDic[row]
            else:
                col[i] = xCategory;
                strDic[row] = xCategory
                xCategory += 1;

    return col

def ConvertStrToNumMatrix(readFileName, writeFileName):
    targetMatrix = pd.read_csv(readFileName).values

    for colNum in range(targetMatrix.shape[1]):
        targetMatrix[:, colNum] = ConvertStrToNumColumn(targetMatrix[:, colNum])

    np.savetxt(writeFileName, targetMatrix)

ConvertStrToNumMatrix('train.csv', 'trainNum.txt')
ConvertStrToNumMatrix('test.csv', 'testNum.txt')


