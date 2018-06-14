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
import time
import operator
from collections import Counter
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

APIKey = 'RGAPI-4c2b92e7-7ec6-42a6-beb7-e475b35c1476'
region = 'NA1'
summonerId = '569095'
n_matchs = 100
TIME_PAUSE = 0.2
TIME_PAUSE_MATCH = 0.3

regions = ['BR1','NA1','KR','RU','OC1','EUN1','EUW1','TR1','LA1','LA2']


def requestLeague(region):
    URL = 'https://{}.api.riotgames.com/lol/league/v3/challengerleagues/by-queue/RANKED_SOLO_5x5?api_key={}'.format(region,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE)
    return response.json()   


def requestMatch(matchID,region):
    URL = 'https://{}.api.riotgames.com/lol/match/v3/matches/{}?api_key={}'.format(region,matchID,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE_MATCH)
    return response.json()

def requestMatchList(accountId,region):
    URL = 'https://{}.api.riotgames.com/lol/match/v3/matchlists/by-account/{}?endIndex={}&api_key={}'.format(region,accountId,n_matchs,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE)
    return response.json()

def requestSummoner(summonerId,region):
    URL = 'https://{}.api.riotgames.com/lol/summoner/v3/summoners/{}?api_key={}'.format(region,summonerId,APIKey)
    response = requests.get(URL)
    time.sleep(TIME_PAUSE)
    return response.json()

### Busca Jogadores de acordo com a lista de Ids
def BuscarJogadores(ProPlayers,qtd_jogadores,region):
    print('#####Buscando Players#######')
    i=0
    for player in ProPlayers.entries:
        try:
            ResponseJSON = requestSummoner(player['playerOrTeamId'],region)
            print('Player {} --- {}\n'.format(ResponseJSON['name'],i))
            if i == 0:
                SummonersList = pd.DataFrame([ResponseJSON])
            else:
                SummonersList= SummonersList.append([ResponseJSON])
            i = i + 1
            if i >= qtd_jogadores:
                break;
        except Exception as p:
            print(p)
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
            print('Lista de Partidas {}\n'.format(i))
        except Exception as p:
            print(p)
            pass
    return MatchList

def BuscaPartidas(MatchList, queue = 420, qtd_max = 100000, region = 'BR1'):
    i = 0
    for match_list in MatchList.matches:
        try:
            for match in match_list:
                try:
                    ResponseJSON = requestMatch(match['gameId'],region)
                    print('Partida {}\n'.format(i))
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
                print ('{} : {}'.format(i,qtd_max))
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



#def BuscarJogadores2(ProPlayers,qtd_jogadores):
#    print('#####Buscando Players#######')
#    i=0
#    entries_max = 9999
#    size = len(ProPlayers)
#    ####Pegar a menor fila ###############
#    for i in range(size):
#        if(len(ProPlayers.iloc[i].entries) < entries_max ):
#            entries_max = len(ProPlayers.iloc[i].entries)
#    print(entries_max)
#    Contador = 0
#    k=0
#    try:
#        for k in range(entries_max):
#            for i in range(size):
#                ResponseJSON = requestSummoner(ProPlayers.iloc[i]['entries'][k]['playerOrTeamId'],ProPlayers.iloc[i]['region'])
#                ResponseJSON['region'] = ProPlayers.iloc[i]['region']
#                print('Player {} --- {} ----Region {}\n'.format(ResponseJSON['name'],Contador,ResponseJSON['region']))
#                if Contador == 0:
#                    List = pd.DataFrame([ResponseJSON])
#                else:
#                    List= List.append([ResponseJSON])
#                Contador = Contador + 1
#                if Contador >= qtd_jogadores:
#                    break;
#            if Contador >= qtd_jogadores:
#                break;
#    except Exception as p:
#        print(p)
#        pass
#    return List    
    
    
#    for player in ProPlayers.entries:
#        try:
#            ResponseJSON = requestSummoner(player['playerOrTeamId'],player['region'])
#            ResponseJSON['region'] = player['region']
#            print('Player {} --- {} ----Region {}\n'.format(ResponseJSON['name'],i,ResponseJSON['region']))
#            if i == 0:
#                SummonersList = pd.DataFrame(ResponseJSON)
#            else:
#                SummonersList= SummonersList.append(ResponseJSON)
#            i = i + 1
#            if i >= qtd_jogadores:
#                break;
#        except Exception as p:
#            print(p)
#            pass
#    return SummonersList


#def ListadePartidas2(SummonersList):
#    i=0
#    print('#####Buscando Lista de Partidas#######')    
#    for x in range(len(SummonersList)):
#        try:
#            ResponseJSON = requestMatchList(SummonersList.accountId[x],SummonersList.region[x])
#            ResponseJSON['region'] = SummonersList.region[x]
#            if i == 0:
#                MatchList = pd.DataFrame([ResponseJSON])
#            else:
#                MatchList = MatchList.append([ResponseJSON])
#            i = i+ 1
#            print('Lista de Partidas {}\n'.format(i))
#        except Exception as p:
#            print(p)
#            pass
#    return MatchList



#def DataFramePartidas(MatchList):
#    
#    Dict_Partida = {}
#    size_lista_partidas = len(MatchList.matches)
#    size_qtd_partidas = len(MatchList.matches[0]) 
#    Contador = 0
#    try:
#        for i in range(size_qtd_partidas):
#            for x in range(size_lista_partidas):
#                Dict_Partida['gameId'] = MatchList.matches[x][i]['gameId']
#                Dict_Partida['platformId'] = MatchList.matches[x][i]['platformId']
#                if Contador == 0:
#                    Partidas = pd.DataFrame([Dict_Partida])
#                else:
#                    Partidas = Partidas.append([Dict_Partida]) 
#                Contador = Contador + 1
#    except Exception as p:
#        print(p)
#        pass
#    return Partidas



#def BuscaPartidas2(MatchList, queue = 420, qtd_max = 100000):
#    x =0
#    for i in range(len(MatchList)):
#        try:
#            ResponseJSON = requestMatch(MatchList.iloc[i]['gameId'],MatchList.iloc[i]['platformId'])
#            print('Partida {}\n'.format(i))
#            if ResponseJSON['queueId'] == queue:    
#                Dict_Partida = {}
#                Dict_Partida['gameId'] = ResponseJSON['gameId']
#                for Team in range(2):
#                    Dict_Partida['Time_'+str(Team)+'_Result'] = ResponseJSON['teams'][Team]['win']
#                    for Player in range(5):
#                          Dict_Partida['_Player_'+'T'+str(Team)+str(Player)+'_Ban_Champion'] = ResponseJSON['teams'][Team]['bans'][Player]['championId']
#                for Player in range(10):
#                    Dict_Partida['Player_'+str(Player)+'_SummonerId'] = ResponseJSON['participantIdentities'][Player]['player']['summonerId']
#                    Dict_Partida['Player_'+str(Player)+'_Champion'] = ResponseJSON['participants'][Player]['championId']
#                    Dict_Partida['Player_'+str(Player)+'_Lane'] = ResponseJSON['participants'][Player]['timeline']['lane']
#                if i == 0:
#                    Matchs = pd.DataFrame([Dict_Partida])
#                else:
#                   Matchs = Matchs.append([Dict_Partida])
#                i= i+1
#                x= x+1
#                if x >= 3000:
#                    Matchs.to_csv('Database\Matchs\Matchs_'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M"))+'.csv', sep =';')
#                    x=0
#        except Exception as p:
#            print(p)
#            pass
#        print ('{} : {}'.format(i,qtd_max))
#        if i > qtd_max:
#            break;
#    return Matchs

#def GerarBase2(qtdPartidas = 999999, qtd_jogadores = 999999):
#    ### Carrega Lista de Jogadores do Challenger
#    List_Summoners = []
#    for region in regions:
#        ResponseJSON = requestLeague(region)
#        SummonersList = pd.DataFrame([ResponseJSON])
#        SummonersList['region'] = region
#        List_Summoners.append(SummonersList)
#    SummonersList = pd.concat(List_Summoners)    
#    SummonersList = shuffle(SummonersList)
#    SummonersList.reset_index(inplace=True)
#    qtd_jogadores = 1000
#    ### Carrega Lista de Summoners através da lista de jogadores
#    SummonersList_id = BuscarJogadores2(SummonersList,1000)
#    SummonersList_id.reset_index(inplace=True)
#    ### Carrega Lista de Partidas dos Jogadores
#    MatchList = ListadePartidas2(SummonersList_id)
#    ### Carrega Partidas
#    MatchList.reset_index(inplace=True)
#    Partidas_List = DataFramePartidas(MatchList)
#    Partidas_List = shuffle(Partidas_List)
#    Partidas_List.to_pickle('Partidas_List.pickle')
#    qtdPartidas = 9999999
#    Matchs = BuscaPartidas2(Partidas_List,420,qtdPartidas)
#    ### Grava Partidas
#    Matchs.to_csv('Database\Matchs\Matchs_'+str(datetime.datetime.now().strftime("%Y-%m-%d-%H_%M"))+'.csv', sep =';')
#    return Matchs


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
    

def main():
regions = ['BR1','NA1','KR','RU','OC1','EUN1','EUW1','TR1','LA1','LA2']
    Matchs = GerarBase(15000,99999, 'LA2')
    Matchs = AbrirBases()
    MelhoresChamps(Matchs, 5)
    
    Database = Matchs.iloc[:, 1:32]
    drop_list = [x for x in Matchs.columns if 'SummonerId' in x]
    Database.drop(columns=drop_list, inplace= True)
    Database['Time_0_Result'].replace(to_replace = 'Win', value = '1', inplace = True) 
    Database['Time_0_Result'].replace(to_replace = 'Fail', value = '0', inplace = True)
    X = Database.iloc[:,:-1]
    Y = Database.iloc[:,20]
    X = X.apply(LabelEncoder().fit_transform)
    onehotencoder = OneHotEncoder()
    X = onehotencoder.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
    
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout
    model = Sequential()
    model.add(Dense(100, input_dim=1460, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(600, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
    
    y_pred = model.predict_classes(X_test,batch_size=128, verbose=0)
    cm = confusion_matrix(y_test,y_pred)
    print('Accuracy Test : {}'.format((cm[0][0]+cm[1][1])/sum(sum(cm))))
    
    
    
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
#    model.fit(X_train, y_train, epochs=100, batch_size=32)
#    loss_and_metrics = model.evaluate(X_test, y_test, batch_size=128)
#    
#    y_pred = model.predict_classes(X_test,batch_size=128, verbose=0)
#    cm = confusion_matrix(y_test,y_pred)

