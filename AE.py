import pandas as pd
import torch
import numpy as np
import pickle
from torch.utils import data
def dataInPath():
    return ''
def modelOutPath():
    return ''
def path1():
    return ''
class Dataset(torch.utils.data.Dataset):
  def __init__(self, array1,target1):
        self.data = array1
        self.target = target1
        

  def __len__(self):
        return self.data.shape[0]

  def __getitem__(self, index):
        X = self.data[[index],:]
        Y=self.target[[index],:]
        return X,Y

class AE(torch.nn.Module):
    def __init__(self,inputDim):
        super().__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inputDim, 67),
            torch.nn.ReLU(),
            torch.nn.Linear(67, 67),
            # torch.nn.ReLU(),
            # torch.nn.Linear(67, 67)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(67, 67),
            torch.nn.ReLU(),
            torch.nn.Linear(67, 67),
            # torch.nn.ReLU(),
            # torch.nn.Linear(67, 67),
            torch.nn.Sigmoid()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = 2*self.decoder(encoded)
        return decoded,encoded

def runTrain(model,maxEpochs,trGen,valGen,device,criterion,optimizer,modelName):
    minValLoss=1000
    trLosses,valLosses=[],[]
    for epoch in range(maxEpochs):
        tempTrLoss=[]
        for local_batch,local_outputs in trGen:
            local_batch,local_outputs = local_batch.to(device),local_outputs.to(device)
            # print(local_batch.shape,local_outputs.shape)
            optimizer.zero_grad()
            preds,encoded=model.forward(local_batch)
            # print(local_batch.shape,local_outputs.shape,preds.shape)
            trLoss=criterion(preds,local_outputs)
            tempTrLoss.append(trLoss.detach().numpy())
            # print(trLoss.detach().numpy())
            trLoss.backward()
            optimizer.step()
        trLosses.append(np.mean(tempTrLoss))
        with torch.set_grad_enabled(False):
            tempValLoss=[]
            for local_batch,local_outputs in valGen:
                local_batch,local_outputs = local_batch.to(device),local_outputs.to(device)
                preds,encoded=model.forward(local_batch)
                valLoss=criterion(preds,local_outputs)
                tempValLoss.append(np.mean(valLoss.detach().numpy()))
            meanValLoss=np.mean(tempValLoss)
            # print(meanValLoss)
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
    learningRate=1e-4
    maxEpochs,batchSize=100,5
    dictLosses={'TrainLosses':{},'ValidationLosses':{}}
    partition=pickle.load(open(dataInPath()+'partition1.pkl','rb'))
    dictSNPsToInclude=pickle.load(open(dataInPath()+'dictSNPsToInclude.pkl','rb'))
    
    for seed,dict1 in dictSNPsToInclude.items():
        iter0=0
        dictPreds={}
        listSNPs0=dict1[67]
        tarTr,_,tarVal,_,tarTs,_,tarTs2,_=returnTrValTest(partition,listSNPs0)
        tarTr,tarVal,tarTs,tarTs2=setNANsToZeros(tarTr,tarVal,tarTs,tarTs2)
        for snpNum,listSNPs in dict1.items():
            arrTr,_,arrVal,_,arrTs,_,arrTs2,_=returnTrValTest(partition,listSNPs)
            arrTr,arrVal,arrTs,arrTs2=setNANsToZeros(arrTr,arrVal,arrTs,arrTs2)
            trainGen=returnGenerator(arrTr,tarTr,{'batch_size':batchSize,'shuffle': True})
            valGen=returnGenerator(arrVal,tarVal,{'batch_size':batchSize,'shuffle': False})
            ae1=AE(len(listSNPs))
            optimizer = torch.optim.Adam(ae1.parameters(), lr=learningRate)
            criterion=torch.nn.MSELoss()
            modelName='ae'+seed+'Dim'+str(len(listSNPs))
            model,trLosses,valLosses=runTrain(ae1,maxEpochs,trainGen,valGen,device,criterion,optimizer,modelName)
            trainLosses,validationLosses=dictLosses['TrainLosses'],dictLosses['ValidationLosses']
            
            trainLosses[modelName],validationLosses[modelName]=trLosses,valLosses
            dictLosses['TrainLosses']=trainLosses
            dictLosses['ValidationLosses']=validationLosses
            print(len(listSNPs))
            dictInfo={'dictLosses':dictLosses,'LearningRate':learningRate,
                      'batchSize':batchSize,'numEpochs':maxEpochs}
            pickle.dump(dictInfo, open(modelOutPath()+modelName+'.pkl', "wb"))
    






