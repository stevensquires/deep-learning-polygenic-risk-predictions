import pandas as pd
import torch
import numpy as np
import pickle
from torch.utils import data
def dataInPath():
    return ''
def path1():
    return ''
def modelOutPath():
    return ''
class Dataset(torch.utils.data.Dataset):
  def __init__(self, array1,tar1):
        self.data = array1
        self.target=tar1

  def __len__(self):
        return self.data.shape[0]

  def __getitem__(self, index):
        X = self.data[[index],:]
        y=self.target[[index]]
        return X,y


class MLP(torch.nn.Module):
    def __init__(self,inputDim):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inputDim, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

def runTrain(model,maxEpochs,trGen,valGen,device,criterion,optimizer,modelName):
    minValLoss=2000
    trLosses,valLosses=[],[]
    for epoch in range(maxEpochs):
        tempTrLoss=[]
        for local_batch,local_outputs in trGen:
            local_batch,local_outputs = local_batch.to(device),local_outputs.to(device).squeeze()
            optimizer.zero_grad()
            preds=model.forward(local_batch).squeeze()
            trLoss=criterion(preds,local_outputs)
            tempTrLoss.append(trLoss.detach().numpy())
            trLoss.backward()
            optimizer.step()
        trLosses.append(np.mean(tempTrLoss))
        with torch.set_grad_enabled(False):
            tempValLoss=[]
            for local_batch,local_outputs in valGen:
                local_batch,local_outputs = local_batch.to(device),local_outputs.to(device).squeeze()
                preds=model.forward(local_batch).squeeze()
                valLoss=criterion(preds,local_outputs)
                tempValLoss.append(np.mean(valLoss.detach().numpy()))
            meanValLoss=np.mean(tempValLoss)
            valLosses.append(meanValLoss)
            if meanValLoss<minValLoss:
                minValLoss=meanValLoss
                torch.save(model.state_dict(),modelOutPath()+modelName+'.params')
    return model,trLosses,valLosses

    

def returnTrValTest(partition,listSNPs):
    # p=len(listSNPs)
    table1=pd.read_csv(path1()+'Combined1.csv',index_col=0)
    table2=table1[listSNPs]
    trList,valList,tsList,tsList2=partition['Train'],partition['Validation'],partition['Test1'],partition['Test2']
    # target1=pickle.load(open(path1()+'grs2Scores.pkl','rb'))
    targets1=pd.read_csv(path1()+'grs2Scores.csv',index_col=0)
    trIn,trTar=table2.loc[trList],targets1.loc[trList]
    valIn,valTar=table2.loc[valList],targets1.loc[valList]
    tsIn,tsTar=table2.loc[tsList],targets1.loc[tsList]
    tsIn2,tsTar2=table2.loc[tsList2],targets1.loc[tsList2]
    return trIn.values,trTar.values,valIn.values,valTar.values,tsIn.values,tsTar.values,tsIn2.values,tsTar2.values
def setNANsToZeros(xTr,xVal,xTs,xTs2):
    xTr,xVal=np.nan_to_num(xTr),np.nan_to_num(xVal)
    xTs,xTs2=np.nan_to_num(xTs),np.nan_to_num(xTs2)
    return xTr,xVal,xTs,xTs2
def returnGenerator(arrIn,tar1,params):
    set1=Dataset(torch.from_numpy(arrIn.astype(np.single)),torch.from_numpy(tar1.astype(np.single)))
    generator = data.DataLoader(set1, **params)
    return generator

if __name__=='__main__':
    device = torch.device("cpu")
    learningRate=5e-5
    maxEpochs,batchSize=100,5
    dictLosses={'TrainLosses':{},'ValidationLosses':{}}
    partition=pickle.load(open(dataInPath()+'partition1.pkl','rb'))
    dictSNPsToInclude=pickle.load(open(dataInPath()+'dictSNPsToInclude.pkl','rb'))
    for seed,dict1 in dictSNPsToInclude.items():
        iter0=0
        dictPreds={}
        for snpNum,listSNPs in dict1.items():
            xTr,yTr,xVal,yVal,xTs,yTs,xTs2,yTs2=returnTrValTest(partition,listSNPs)
            xTr,xVal,xTs,xTs2=setNANsToZeros(xTr,xVal,xTs,xTs2)
            trainGen=returnGenerator(xTr,yTr,{'batch_size':batchSize,'shuffle': False})
            valGen=returnGenerator(xVal,yVal,{'batch_size':batchSize,'shuffle': False})
            mlp1=MLP(len(listSNPs))
            optimizer = torch.optim.Adam(mlp1.parameters(), lr=learningRate)
            criterion=torch.nn.MSELoss()
            modelName='mlp'+seed+'Dim'+str(len(listSNPs))
            model,trLosses,valLosses=runTrain(mlp1,maxEpochs,trainGen,valGen,device,criterion,optimizer,modelName)
            trainLosses,validationLosses=dictLosses['TrainLosses'],dictLosses['ValidationLosses']
            
            trainLosses[modelName],validationLosses[modelName]=trLosses,valLosses
            dictLosses['TrainLosses']=trainLosses
            dictLosses['ValidationLosses']=validationLosses
            print(len(listSNPs))
            pickle.dump(dictLosses, open(modelOutPath()+'MLP'+seed+'losses.pkl', "wb"))
        
    


























