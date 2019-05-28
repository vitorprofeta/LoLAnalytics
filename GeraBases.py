# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:56:44 2018

@author: vitorprofeta
"""
import requests
import pandas as pd
import numpy as np
import time
import datetime
from collections import Counter
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.callbacks import TensorBoard
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from keras.constraints import maxnorm
from keras.optimizers import SGD
from threading import Thread
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import ModelCheckpoint
from sklearn.externals import joblib

APIKey = 'RGAPI-76d4ad23-73a4-4ec4-978b-b42a295c189b'
region = 'NA1'
summonerId = '569095'
n_matchs = 30
TIME_PAUSE = 0.2
TIME_PAUSE_MATCH = 0.3

regions = ['BR1','NA1','KR','RU','OC1','EUN1','EUW1','TR1','LA1','LA2']


def requestLeague(region):
    URL = 'https://{}.api.riotgames.com/lol/league/v4/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key={}'.format(region,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE)
    return response.json()   


def requestMatch(matchID,region):
    URL = 'https://{}.api.riotgames.com/lol/match/v4/matches/{}?api_key={}'.format(region,matchID,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE_MATCH)
    return response.json()

def requestMatchList(accountId,region):
    URL = 'https://{}.api.riotgames.com/lol/match/v4/matchlists/by-account/{}?endIndex={}&api_key={}'.format(region,accountId,n_matchs,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE)
    return response.json()

def requestSummoner(summonerId,region):
    URL = 'https://{}.api.riotgames.com/lol/summoner/v4/summoners/{}?api_key={}'.format(region,summonerId,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE)
    return response.json()

### Busca Jogadores de acordo com a lista de Ids
def BuscarJogadores(ProPlayers,qtd_jogadores,region):
    print('#####Buscando Players#######')
    i=0
    for player in ProPlayers.entries:
        try:
            ResponseJSON = requestSummoner(player['summonerId'],region)
            print('Player {} --- {} --- Region: {}\n'.format(ResponseJSON['name'],i,region))
            if i == 0:
                SummonersList = pd.DataFrame([ResponseJSON])
            else:
                SummonersList= SummonersList.append([ResponseJSON])
            i = i + 1
            if i >= qtd_jogadores:
                break;
        except Exception as p:
            print('Exception: {}'.format(p))
            pass
    return SummonersList

def ListadePartidas(SummonersList,region):
    i=0
    print('#####Buscando Lista de Partidas#######')    
    for accountId in SummonersList.accountId:
        try:
            ResponseJSON = requestMatchList(accountId,region)
            if i == 0:
                MatchList = pd.DataFrame([ResponseJSON])
            else:
                MatchList = MatchList.append([ResponseJSON])
            i = i+ 1
            print('Lista de Partidas: {} -- Region: {}\n'.format(i,region))
        except Exception as p:
            print(p)
            pass
    MatchList.reset_index(inplace=True, drop=True)
    return MatchList

def BuscaPartidas(MatchList, queue = 420, qtd_max = 100000, region = 'BR1'):
    i = 0
    for match_list in MatchList.matches:
        try:
            for match in match_list:
                try:
                    ResponseJSON = requestMatch(match['gameId'],region)
                    print('Partida {} -- Region -- {}\n'.format(i,region))
                    if ResponseJSON['queueId'] == queue:    
                        Dict_Partida = {}
                        Dict_Partida['gameId'] = ResponseJSON['gameId']
                        for Team in range(2):
                            Dict_Partida['Time_'+str(Team)+'_Result'] = ResponseJSON['teams'][Team]['win']
                            for Player in range(5):
                                  Dict_Partida['_Player_'+'T'+str(Team)+str(Player)+'_Ban_Champion'] = ResponseJSON['teams'][Team]['bans'][Player]['championId']
                        for Player in range(10):
                            Dict_Partida['Player_'+str(Player)+'_SummonerId'] = ResponseJSON['participantIdentities'][Player]['player']['summonerId']
                            Dict_Partida['Player_'+str(Player)+'_Champion'] = ResponseJSON['participants'][Player]['championId']
                            Dict_Partida['Player_'+str(Player)+'_Lane'] = ResponseJSON['participants'][Player]['timeline']['lane']
                        if i == 0:
                            Matchs = pd.DataFrame([Dict_Partida])
                        else:
                           Matchs = Matchs.append([Dict_Partida])
                        i= i+1
                except Exception as p:
                    print(p)
                    pass
                print ('{} : {} : {}\n'.format(i,qtd_max,region))
                if i > qtd_max:
                    break;
        except Exception as p:
            print(p)
            pass
        if i > qtd_max:
            break;
    return Matchs

def GerarBase(qtdPartidas = 999999, qtd_jogadores = 999999, region= 'BR1'):
    ### Carrega Lista de Jogadores do Challenger
    ResponseJSON = requestLeague(region)
    ### Carrega Lista de Summoners através da lista de jogadores
    SummonersList = BuscarJogadores(pd.DataFrame(ResponseJSON),qtd_jogadores,region)
    ### Grava a lista de Jogadores
    SummonersList.to_csv('Database\SummonerList\SummonerList'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M"))+'_'+ region + '.csv', sep=';')
    ### Carrega Lista de Partidas dos Jogadores
    MatchList = ListadePartidas(SummonersList,region)
    ### Grava Lista de Partidas
    MatchList.to_csv('Database\MatchList\MatchList'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M"))+'_'+ region + '.csv', sep=';')   
    ### Carrega Partidas
    Matchs = BuscaPartidas(MatchList,420,qtdPartidas,region)
    ### Grava Partidas
    Matchs.to_csv('Database\Matchs\Matchs_'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M"))+ region + '.csv', sep =';')
    return Matchs        

def AbrirBases():
    frames = []
    PATH = 'Database\Matchs'
    for file in os.listdir(PATH):
        frames.append(pd.read_csv(PATH+ "\\"+ file,sep=';'))
    return pd.concat(frames)


def MelhoresChamps(Matchs, qtd =5):
    Time0 = ['Player_{}_Champion'.format(x) for x in range(5)]
    Time1 = ['Player_{}_Champion'.format(x+5) for x in range(5)]
   
    Vitorias = Counter()
    for x in range(5):
        Vitorias = Vitorias + Counter(Matchs[Matchs.Time_0_Result == 'Win'][Time0[x]])
        Vitorias = Vitorias + Counter(Matchs[Matchs.Time_1_Result == 'Win'][Time1[x]])
        
    
    Total_Games = Counter()
    for x in range(5):
        Total_Games = Total_Games + Counter(Matchs[Time0[x]])
        Total_Games = Total_Games + Counter(Matchs[Time1[x]])
    
    Champions = pd.read_json('Champions.json')
    
    for x in range(len(Champions['data'])):
        for key_p in Vitorias:
            if Champions['data'][x]['id'] == key_p:
                Vitorias[Champions['data'][x]['key']] = Vitorias[key_p]
                del Vitorias[key_p]
    
    for x in range(len(Champions['data'])):
        for key_p in Total_Games:
            if Champions['data'][x]['id'] == key_p:
                Total_Games[Champions['data'][x]['key']] = Total_Games[key_p]
                del Total_Games[key_p]
    
    percent_victorys = Counter()    
    for key_t in Total_Games:
        for key_v in Vitorias:
            if key_v == key_t:
                 percent_victorys[key_v] = Vitorias[key_v]/Total_Games[key_v]
    print('\n Champions com Maior número de Jogos \n')
    print(Total_Games.most_common(qtd))
    print('\n Champions com Maior número de Vitórias \n')
    print(Vitorias.most_common(qtd))
    print('\n Champions com Maior percentual de Vitórias \n')
    print(percent_victorys.most_common(qtd))


# baseline
def create_model(): 
    # Criar arquitetura do modelo
    model = Sequential()
    model.add(Dense(100, input_dim=1490, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(450, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(450, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def Train_model(model,X,Y,epochs):
    #Gerar Logs para o TensorBoard
    tbCallBack = keras.callbacks.TensorBoard(log_dir='C:\\logs\\', histogram_freq=0, write_graph=True, write_images=True)
    checkpointer = ModelCheckpoint(filepath="train_weights.hdf5", verbose=1, save_best_only=False)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, callbacks=[tbCallBack,checkpointer])
    return model,X_test, y_test

def TransformSplitMatchs(Matchs):
    Database = Matchs.iloc[:, 1:32]
    drop_list = [x for x in Matchs.columns if 'SummonerId' in x]
    Database.drop(columns=drop_list, inplace= True)
    Database['Time_0_Result'].replace(to_replace = 'Win', value = '1', inplace = True) 
    Database['Time_0_Result'].replace(to_replace = 'Fail', value = '0', inplace = True)
    X = Database.iloc[:,:-1]
    Y = Database.iloc[:,20]   
    return X,Y

def Transform(X, save= False):
    if save:
        labelencoder_x = LabelEncoder()
        X.iloc[:,1] = labelencoder_x.fit_transform(X.iloc[:,1])
        for x in range(3,21,2):   
            X.iloc[:,x] = labelencoder_x.transform(X.iloc[:,x])
        onehotencoder = OneHotEncoder(sparse = False)
        X = onehotencoder.fit_transform(X)
        joblib.dump(labelencoder_x, 'encoder.pkl') 
        joblib.dump(onehotencoder, 'onehotencoder.pkl') 
    else:
        labelencoder_x = joblib.load('encoder.pkl')
        for x in range(1,21,2): 
            X.iloc[:,x] = labelencoder_x.transform(X.iloc[:,x])
        onehotencoder = joblib.load('onehotencoder.pkl')
        X = onehotencoder.transform(X)
    return X


def Champions_Transform(Matchs):
    Champion_dicts = joblib.load('champions_dict.pkl')
    for column in Matchs.columns:
        Matchs[column] = Matchs[column].astype(str)
    return Matchs

def CarregarModelo():
    model = create_model()
    model.load_weights('train_weights.hdf5')
    return model

def main():

    regions = ['BR1','NA1','KR','RU','OC1','EUN1','EUW1','TR1','LA1','LA2']
#    Matchs = GerarBase(5000,99999, region)
    List_Thread = []
    for region in regions:
        List_Thread.append(Thread(target=GerarBase,args=[5000,99999,region]))
    for thread in List_Thread:
        thread.start()
    Matchs = AbrirBases()    
    MelhoresChamps(Matchs, 20)
    X,Y= TransformSplitMatchs(Matchs)    
    X = Transform(X,True)
    model = create_model()
    model,X_test, y_test = Train_model(model,X,Y,10)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)    
    y_pred = model.predict_classes(X_test,batch_size=128, verbose=0)
    y_test = y_test.astype(int)
    cm = confusion_matrix(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print(report)
    print('Accuracy Test : {}'.format((cm[0][0]+cm[1][1])/sum(sum(cm))))
    print(loss_and_metrics)
#    seed = 5
#    np.random.seed(seed)
#    estimators = []
#    estimators.append(('standardize', StandardScaler()))
#    estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=20, batch_size=16, verbose=0)))
#    pipeline = Pipeline(estimators)
#    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
#    results = cross_val_score(pipeline, X, Y, cv=kfold)
#    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#    import h5py    
#    import numpy as np    
#    model = h5py.File('kerasmodel.h5','r') 
#    
    
#    from keras.models import Sequential
#    from keras.layers import Dense
#    model = Sequential()
#    model.add(Dense(100, input_dim=1460, activation='relu'))
#    model.add(Dense(450, activation='relu'))
#    model.add(Dense(450, activation='relu'))
#    model.add(Dense(1, activation='sigmoid'))
#    # Compile model
#    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#    # Fit the model
#    model.fit(X_train, y_train, epochs=5, batch_size=32)
#    # Fit the model
#    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
#    model.fit(X_train, y_train, epochs=100, batch_size=32)
#    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
#    new_model = keras.models.model_from_json(json, custom_objects={})
#    
#    y_pred = model.predict_classes(X_test,batch_size=128, verbose=0)
#    cm = confusion_matrix(y_test,y_pred)

