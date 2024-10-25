import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from joblib import dump
def dataInPath():
    return ''
def path1():
    return ''
def modelOutPath():
    return ''

def returnTrValTest(partition,listSNPs):
    table1=pd.read_csv(path1()+'Combined1.csv',index_col=0)
    table2=table1[listSNPs]
    trList,valList,tsList,tsList2=partition['Train'],partition['Validation'],partition['Test1'],partition['Test2']
    targets1=pd.read_csv(path1()+'grs2Scores.csv',index_col=0)
    trIn,trTar=table2.loc[trList],targets1.loc[trList]
    valIn,valTar=table2.loc[valList],targets1.loc[valList]
    tsIn,tsTar=table2.loc[tsList],targets1.loc[tsList]
    tsIn2,tsTar2=table2.loc[tsList2],targets1.loc[tsList2]
    return trIn.values,trTar.values,valIn.values,valTar.values,tsIn.values,tsTar.values,tsIn2.values,tsTar2.values
def returnErrArrays(numLambda1,numLambda2):
    trErrs=np.zeros((numLambda1,numLambda2),float)
    valErrs=np.zeros((numLambda1,numLambda2),float)
    testErrs=np.zeros((numLambda1,numLambda2),float)
    return trErrs,valErrs,testErrs
def returnErrResults(model,xVals,targets1):
    preds1=model.predict(xVals)
    mse1=round(mse(targets1,preds1),2)
    return mse1,preds1
def returnErrArrays2(num1):
    errs1=np.zeros((num1),float)
    errs2,errs3,errs4=errs1.copy(),errs1.copy(),errs1.copy()
    return errs1,errs2,errs3,errs4
def setNANsToZeros(xTr,xVal,xTs,xTs2):
    xTr,xVal=np.nan_to_num(xTr),np.nan_to_num(xVal)
    xTs,xTs2=np.nan_to_num(xTs),np.nan_to_num(xTs2)
    return xTr,xVal,xTs,xTs2
    

if __name__=='__main__':
    partition=pickle.load(open(dataInPath()+'partition1.pkl','rb'))
    dictSNPsToInclude=pickle.load(open(dataInPath()+'dictSNPsToInclude.pkl','rb'))
    for seed,dict1 in dictSNPsToInclude.items():
        trErrs2,valErrs2,testErrs2,testErrs2=returnErrArrays2(len(dict1))
        iter0=0
        dictPreds={}
        for snpNum,listSNPs in dict1.items():
            numLambda1,numLambda2=5,5
            lambda1Vals=np.logspace(-8,1,num=numLambda1)
            lambda2Vals=np.logspace(-8,1,num=numLambda2)
            trErrs,valErrs,testErrs=returnErrArrays(numLambda1,numLambda2)
            xTr,yTr,xVal,yVal,xTs,yTs,xTs2,yTs2=returnTrValTest(partition,listSNPs)
            xTr,xVal,xTs,xTs2=setNANsToZeros(xTr,xVal,xTs,xTs2)
            model2=LinearRegression().fit(xTr,yTr)
            dump(model2,modelOutPath()+'LinReg'+seed+'Dim'+str(len(listSNPs))+'.joblib') 
            trErrs2[iter0],trPreds=returnErrResults(model2,xTr,yTr)
            valErrs2[iter0],valPreds=returnErrResults(model2,xVal,yVal)
            testErrs2[iter0],testPreds=returnErrResults(model2,xTs,yTs)
            predsTs2=model2.predict(xTs2)
            dictTemp={'Train':[yTr,trPreds],'Validation':[yVal,valPreds],'Test':[yTs,testPreds],
                      'Test2':[yTs2,predsTs2]}
            dictPreds[len(listSNPs)]=dictTemp
            dictLosses={'TrainLosses':trErrs2,'ValidationLosses':valErrs2,'TestLosses':testErrs2}
            pickle.dump(dictLosses,open(modelOutPath()+'LinReg'+seed+'losses.pkl', "wb"))
            pickle.dump(dictPreds,open(modelOutPath()+'LinReg'+seed+'preds.pkl', "wb"))
            iter0+=1
            print(iter0)



