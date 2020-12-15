# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import random
import matplotlib as plt
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer
#pd.set_option("display.precision", 5)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold,StratifiedShuffleSplit,cross_val_score,cross_validate
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix,average_precision_score,plot_precision_recall_curve,classification_report,precision_recall_fscore_support
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.preprocessing import label_binarize


# %%
# config
levels = 10
NAME = 'AAPL'
targetEstimator = 'MID.Dir'
feaType = 'SSS'

# %%
# preparation, read in data
name_list = ['AMZN','AAPL','GOOG','INTC','MSFT']

AAPL_msg_df = pd.read_csv('AAPL_message_10_4hrs.csv',names = ['Time (sec)',
    'Event Type',
    'Order ID',
    'Size',
    'Price',
    'Direction'])
AAPL_odr_df = pd.read_csv('AAPL_order_10_4hrs.csv',
                    names = [f'{ty}.{VP}.{i}' 
                        for i in range(1,11) for ty in ['ASK','BID'] 
                            for VP in ['Price','Size'] ])
AAPL_odr_df['Time (sec)'] = AAPL_msg_df['Time (sec)']

# %%
#df_pack = {'AMZN':AMZN_odr_df,'AAPL': AAPL_odr_df, 'GOOG': GOOG_odr_df, 'INTC':INTC_odr_df , 'MSFT': MSFT_odr_df}
df_pack = {NAME: AAPL_odr_df}
msg_pack = {NAME:AAPL_msg_df}
TYPE = ['ASK','BID']
SIZEPRICE = ['Size','Price']

# %%
# name -> time -> b/a -> size/price -> level
# e.g. Basic_set['AMZN'][34200.191994185]['ASK']['Size']['2']
# this is only used in the time insensitive set
Basic_set_whole = dict([(name,
                dict([(time,
                    dict([(ty,
                        dict([(SP,
                            dict([(str(le),
                    df_pack[name]['.'.join([ty,SP,str(le)])].iloc[idx])
                            for le in range(1,levels+1)]))
                                for SP in SIZEPRICE]))
                                    for ty in TYPE]))
                                        for idx, time in enumerate(df_pack[name]['Time (sec)'])]))
                                            for name in list(df_pack)])

# %%
#msg_pack = {'AMZN':AMZN_msg_df,'AAPL': AAPL_msg_df, 'GOOG': GOOG_msg_df, 'INTC':INTC_msg_df, 'MSFT': MSFT_msg_df}

# construct label set across all times
# shape (400391,7)
time_series = list(df_pack[NAME]['Time (sec)']).copy()
time_frame = {'Time (sec)':time_series,'MID.Dir':time_series,'Spread.crossing':time_series}
label_df = pd.DataFrame(time_frame)
label_df['MID'] = (df_pack[NAME]['ASK.Price.1'] + df_pack[NAME]['BID.Price.1'])/2
label_df['MID.Diff'] = label_df['MID'].diff()
number = 5
label_df['Spread.BID.Diff'] = df_pack[NAME]['BID.Price.1'].shift(number) - df_pack[NAME]['ASK.Price.1']
label_df['Spread.ASK.Diff'] = df_pack[NAME]['ASK.Price.1'].shift(number) - df_pack[NAME]['BID.Price.1']
timeLog = list(df_pack[NAME]['Time (sec)'])
def check_mid_price_direction(df):
    if df > 0:
        direction = 1
    elif df < 0:
        direction = -1
    else:
        direction = 0
    return direction
# assign direction based on mid price movement: every row
label_df['MID.Dir'] = label_df['MID.Diff'].apply(check_mid_price_direction)
def check_spread_crossing_direction(val1,val2):
    # suggest num = 200
    if val1 > 0:
        direction = 1
    elif val2 <0:
        direction = -1
    else:
        direction = 0
    return direction

label_df['Spread.crossing'] = label_df.apply(lambda x: check_spread_crossing_direction(x['Spread.BID.Diff'],x['Spread.ASK.Diff']),axis = 1)
label_df = label_df.fillna(0)
label_pack = {NAME:label_df}


# %%
# be careful with this block of code, should start from the label_pack, 
# cannot simply take the previous 2000 record.
# in this simple trading strategy, the author did not implement the stratified k fold.
# this RandomList is time(float)
# select 2000 points
# RandomList = list(np.arange(2000))
# this RandomList is the index(int)

## RandomList = label_pack[NAME].copy()['Time (sec)'].head(2000).drop_duplicates().index



# construct sets
# v1 should indexed by time
def construct_basic_set(df_pack,Basic_set,randomList,name,Types,SizePrice,levels):
    v1 = {}
    for idx, time in enumerate(df_pack[name]['Time (sec)']):
        if idx in randomList:
            # create dictionary for each time point
            v1_une = {}
            for ty in Types:
                for SP in SizePrice:
                    for le in range(1,levels+1):
                        v1_une['_'.join([ty,SP,str(le)])] = Basic_set[name][time][ty][SP][str(le)]
            v1[time] = v1_une
    return v1


def construct_time_insen_set(df_pack,Basic_set,randomList,name,Type,SizePrice,levels):
    v2 = {}
    v3={}
    v4={}
    v5={}
    for idx, time in enumerate(df_pack[name]['Time (sec)']):
        if idx in randomList:
            # create dictionary for each time point
            v2_une = {}
            for le in range(1,levels+1):
                # should be df_pack instead of Basic_set
                v2_une['_'.join(['Price','Spread',str(le)])] = Basic_set[name][time]['ASK']['Price'][str(le)] -                                                 Basic_set[name][time]['BID']['Price'][str(le)]
                v2_une['_'.join(['Price','Mid',str(le)])] = Basic_set[name][time]['ASK']['Price'][str(le)] +                                                 Basic_set[name][time]['BID']['Price'][str(le)]
            v2[time] = v2_une

            v3_une = {}
            v3_une['Price_diff_ask'] =  Basic_set[name][time]['ASK']['Price'][str(levels)] - Basic_set[name][time]['ASK']['Price'][str(1)]
            v3_une['Price_diff_bid'] =  Basic_set[name][time]['BID']['Price'][str(1)] - Basic_set[name][time]['BID']['Price'][str(levels)]
            # shold this be levels +1?
            for le in range(1,levels):
                v3_une['_'.join(['Price','Ask','abs_diff',str(le)])] = abs(Basic_set[name][time]['ASK']['Price'][str(le+1)] -                                                 Basic_set[name][time]['BID']['Price'][str(le)])
                v3_une['_'.join(['Price','Bid','abs_diff',str(le)])] = Basic_set[name][time]['BID']['Price'][str(le+1)] -                                                Basic_set[name][time]['BID']['Price'][str(le)]
            v3[time] = v3_une

            v4_une = {}
            for SP in SizePrice:
                for ty in Type:
                    v4_une['_'.join(['Mean',ty,SP])] = np.mean(list(Basic_set[name][time][ty][SP].values()))
            v4[time] = v4_une

            v5_une = {}
            v5_une['Accum_price_diff'] = np.sum(np.array(list(Basic_set[name][time]['ASK']['Price'].values())) -                                                     np.array(list(Basic_set[name][time]['BID']['Price'].values())))
            v5_une['Accum_size_diff'] =  np.sum(np.array(list(Basic_set[name][time]['ASK']['Size'].values())) -                                                     np.array(list(Basic_set[name][time]['BID']['Size'].values())))
            v5[time] = v5_une
    return v2,v3,v4,v5

# should also pass in msg book
# loop implementation is slow
def construct_time_sensi_set(df_pack,randomList,name,msg_pack):
    v6 = {}
    v7 = {}
    v8 = {}
    v9 = {}
    errorLog = {}
    baseTime = df_pack[name]['Time (sec)'].iloc[0]
    timeLog = list(df_pack[name]['Time (sec)'])
    # what if have multiple same time?
    for idx, time in enumerate(df_pack[name]['Time (sec)']):
        if idx in randomList:
            if time > baseTime + 1:
                v6_une = {}
                closestTime1 = min(timeLog, key=lambda x:abs(x-(time-1)))
                for le in range(1,levels + 1):
                    try:
                        v6_une['Ask_price_derive_' + str(le)] = Basic_set_whole[name][time]['ASK']['Price'][str(le)] -  Basic_set_whole[name][closestTime1]['ASK']['Price'][str(le)]
                        v6_une['Bid_price_derive_' + str(le)] = Basic_set_whole[name][time]['BID']['Price'][str(le)] - Basic_set_whole[name][closestTime1]['BID']['Price'][str(le)]
                        v6_une['Ask_size_derive_' + str(le)] = Basic_set_whole[name][time]['ASK']['Size'][str(le)] -   Basic_set_whole[name][closestTime1]['ASK']['Size'][str(le)]
                        v6_une['Bid_size_derive_' + str(le)] = Basic_set_whole[name][time]['BID']['Size'][str(le)] -   Basic_set_whole[name][closestTime1]['BID']['Size'][str(le)]
                    except KeyError as e:
                        errorLog[time] = str(e)
                v6[time] = v6_une
                
                v7_une = {}
                def generate_indi_based_on_time(v:dict, closestTime):

                    after = msg_pack[name]['Time (sec)'] > closestTime
                    before = msg_pack[name]['Time (sec)'] < time
                    subMsgDf = msg_pack[name][after & before]
                    limitAsk = (subMsgDf['Event Type'] == 1) & (subMsgDf['Direction'] == -1)
                    limitBid = (subMsgDf['Event Type'] == 1) & (subMsgDf['Direction'] == 1)
                    # market order has reverse logic as limit order
                    marketAsk = ((subMsgDf['Event Type'] == 4) | (subMsgDf['Event Type'] == 5)) & (subMsgDf['Direction'] == 1)
                    marketBid = ((subMsgDf['Event Type'] == 4) | (subMsgDf['Event Type'] == 5)) & (subMsgDf['Direction'] == -1)
                    cancelAsk = (subMsgDf['Event Type'] == 2) & (subMsgDf['Direction'] == -1)
                    cancelBid = (subMsgDf['Event Type'] == 2) & (subMsgDf['Direction'] == 1)
                    v['Limit_ask'] = len(subMsgDf[limitAsk])
                    v['Limit_bid'] = len(subMsgDf[limitBid])
                    v['Market_ask'] = len(subMsgDf[marketAsk])
                    v['Market_bid'] = len(subMsgDf[marketBid])
                    v['Cancel_ask'] = len(subMsgDf[cancelAsk])
                    v['Cancel_bid'] = len(subMsgDf[cancelBid])
                    return v

                v7[time] = generate_indi_based_on_time(v7_une, closestTime1)

                # v8 check discrepancy between short term and long term
                # all zeros
                v8_une = {}
                v8_une_temp1 = {}
                v8_une_temp2 = {}
                closestTime10 = min(timeLog, key=lambda x:abs(x-(time-10)))
                closestTime900 = min(timeLog, key=lambda x:abs(x-(time-900)))
                v8_une_short = Counter(generate_indi_based_on_time(v8_une_temp1, closestTime10))
                v8_une_long = Counter(generate_indi_based_on_time(v8_une_temp2, closestTime900))
                v8_une_short.subtract(v8_une_long)
                for key in list(v8_une_short.keys()):
                    if key in ['Limit_ask','Limit_bid','Market_ask','Market_bid']:
                        if v8_une_short[key] > 0:
                            v8_une['Ind_' + key] = 1  
                        else:
                            v8_une['Ind_' + key] = 0
                
                v8[time] = v8_une

                # v9 acceleration of certain trading type
                # why v9 and v7 are the same
                v9_une = {}
                v9_une['acce_limit_ask'] = v7[time]['Limit_ask']
                v9_une['acce_limit_bid'] = v7[time]['Limit_bid']
                v9_une['acce_market_ask'] = v7[time]['Market_ask']
                v9_une['acce_market_bid'] = v7[time]['Market_bid']
                v9[time] = v9_une
    return v6,v7,v8,v9,errorLog
    

def assign_label_to_attributes(vPack,labelSet,istest):
    '''
    This function is time consuming, need to modify.
    based on index
    vPack: a combo of features sets v1 ->v9, if feaType is BF
    label_set: lable of sets index by time
    return : 140 by 1500 df that need to feed in IG calculation
             34202.77068	34209.11439	34209.31355
    ask_size_1
    ask_size_2 ...
     . 
     . 
     . 

    '''
    #label_pack['AMZN'][label_pack['AMZN']['Time (sec)'].isin([34200.01746,34202.77893])]
    if istest:
        dfList = []
        for name,sets in vPack.items():
            temp = pd.DataFrame.from_dict(sets)
            dfList.append(temp[labelSet.index.tolist()])
        attributesAndLabelsDf = pd.concat(dfList)
    else:
        dfList = []
        for name,sets in vPack.items():
            dfList.append(pd.DataFrame.from_dict(sets))
        attributesAndLabelsDf = pd.concat(dfList)
    return attributesAndLabelsDf

# %%
def slice_label_pack(randomList,name,label_pack, spread):
    '''
    metic: Spread, MidPrice
    return a pd.Series sorted by time
    '''
    sortedList = sorted(randomList)
    if spread:
        temp = list(label_pack[name].iloc[sortedList]['Spread.crossing'])
        return pd.Series(temp,index = label_pack[name].iloc[sortedList]['Time (sec)'])
    else:
        temp = list(label_pack[name].iloc[sortedList]['MID.Dir'])
        return pd.Series(temp,index = label_pack[name].iloc[sortedList]['Time (sec)'])


def group_attributes(vPack):
    '''
    return: feature list, then pass into select_attributes_based_on_IG.
    '''
    attributesList = []
    attributesDict = {}
    index = [int(ele.split('v')[-1]) for ele in list(vPack.keys())]
    for i in index:
        attributesList += list(list(vPack[f'v{i}'].values())[0].keys())
    for attri in attributesList:
        attributesDict[attri] = [] 
    for i in index:
        for element in list(vPack[f'v{i}'].values()):
            for k,v in element.items():

                attributesDict[k] += [v]

    return attributesDict,attributesList
# %%
# feature selection
# we have many features x and each feature has its corresponding domain, calculate the IG for this feature
def entropy(target_row):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements,counts = np.unique(target_row,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name="Label"):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataframe for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """    
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data.loc[target_name])
    
    ##Calculate the entropy of the dataset
    
    #Calculate the values and the corresponding counts for the split attribute, e.g. ASK.PRIZE.1
    vals,counts= np.unique(data.loc[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    # df.loc[:,condition on rows]
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.loc[:,data.loc[split_attribute_name]==vals[i]].dropna().loc[target_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

def select_attributes_based_on_IG(data,target_name,featureList):
    '''
    :RETURN : a dictionary sorted by value
    '''
    featureImportance = dict()
    for split_attribute_name in featureList:
        featureImportance[split_attribute_name] = InfoGain(data,split_attribute_name,target_name)
    sortedFeatureImportance = dict(sorted(featureImportance.items(),key=lambda item: item[1],reverse=True))
    return sortedFeatureImportance


# %%
def train_model(df_pack,RandomList,NAME,TYPE,SIZEPRICE,levels,feaType,isSpread):
    '''
    takes in necessary infomation, construct feature sets, train svm model for each individual label in the label set.
    RETURN a dictionary contains 3 svms
    '''
    # use the first 2000 points as the traing set
    #randomList = random.sample(range(1, len(df_pack['AMZN'])), 1500)
    RandomList = label_pack[NAME].copy()['Time (sec)'].loc[RandomList].drop_duplicates().index
    Basic_set = dict([(name,
                    dict([(time,
                        dict([(ty,
                            dict([(SP,
                                dict([(str(le),
                        df_pack[name]['.'.join([ty,SP,str(le)])].iloc[idx])
                                for le in range(1,levels+1)]))
                                    for SP in SIZEPRICE]))
                                        for ty in TYPE]))
                                            for idx, time in enumerate(df_pack[name]['Time (sec)']) if idx in RandomList]))
                                                for name in list(df_pack)])

    v1 = construct_basic_set(df_pack,Basic_set,RandomList,NAME,TYPE,SIZEPRICE,levels)
    v2,v3,v4,v5 = construct_time_insen_set(df_pack,Basic_set,RandomList,NAME,TYPE,SIZEPRICE,levels)
    v6,v7,v8,v9,errorLog = construct_time_sensi_set(df_pack,RandomList,NAME,msg_pack)
    # feature set option
    if feaType == 'BF':
        vPack = {'v1':v1}
    elif feaType == 'BTIS':
        vPack = {'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5}
    elif feaType == 'BTS':
        vPack = {'v1':v1,'v6':v6,'v7':v7,'v8':v8,'v9':v9}
    else:
        vPack = {'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5,'v6':v6,'v7':v7,'v8':v8,'v9':v9}

    # here to choose you use Spread or MidPrice as indicator.
    label_df_slice = slice_label_pack(RandomList,NAME,label_pack,isSpread)
    # it is now has dimension #of features by 1662
    featuresAndLabels = assign_label_to_attributes(vPack,label_df_slice,False)
    featuresAndLabels.loc['Label'] = label_df_slice.copy()
    featuresAndLabels.loc['Label.U'] = label_df_slice.mask(label_df_slice<= 0, -1).copy()
    featuresAndLabels.loc['Label.D'] = label_df_slice.mask(label_df_slice>= 0, 1).copy()
    featuresAndLabels.loc['Label.S'] = label_df_slice.mask(label_df_slice!= 0, 1).mask(label_df_slice== 0, -1).copy()

    featuresDict,featuresList = group_attributes(vPack)

    movementDir = ['U','D','S']
    svcDict = dict()
    finalFeatureDict = dict()
    test_features = featuresAndLabels.copy().replace(np.nan, 0).T
    for label in movementDir:
        #X_train = test_features.drop([f'Label.{label}'],axis = 1)
        X_train = featuresAndLabels.replace(np.nan, 0).T
        # there is no k-fold validation in this step
        Y_train = test_features[f'Label.{label}']
        # preconstruct a OneVsRest model, get the f1 score using all features.
        svc = SVC(C = 0.25, kernel="poly",degree=2,decision_function_shape='ovr')
        f1Score = cross_validate(svc, X_train, Y_train, cv=StratifiedKFold(10), scoring=['f1'],verbose=0)
        f1Score =  np.mean(f1Score['test_f1'])
        # feature selection using IG
        featureRankingDict = select_attributes_based_on_IG(featuresAndLabels,f"Label.{label}",featuresList)
        featureRanking = list(featureRankingDict.keys())
        finalFeatureSet = [featureRanking[0]]
        for feature in featureRanking[1:]:
            updated_X_train = X_train[finalFeatureSet]
            subsetf1Score = cross_validate(svc, updated_X_train, Y_train, cv=StratifiedKFold(10), scoring=['f1'],verbose=0)
            subsetf1Score =  np.mean(subsetf1Score['test_f1'])
            if subsetf1Score > 0.95 * f1Score:
                break
            else:
                finalFeatureSet.append(feature)

        updated_X_train = X_train[finalFeatureSet]
        svc.fit(updated_X_train,Y_train)
        svcDict[label] = svc
        finalFeatureDict[label] = finalFeatureSet
    return svcDict,vPack,finalFeatureDict

#%%
# def simple_trading_strategy(numOfTraining, numOfPredicting,label_pack,NAME,TYPE):
#     '''
#     pass in a dataframe, with dimension 2000 + 4* 10,000 by num of features.
#     train using the first 2000 rows, predict next 10,000,
#     then retrain using the previous  2000 at time point 12,000,
#     note how to slice data is linked through the randomList   
#     :RETURN: tradingResultsDf which consist of the time, trading signal, p&L
#     '''
numOfPredicting = 10000
numOfTraining = 2000
feaType = 'full'
isSpread = True
# should be 4
pAndLdfs = dict()
for i in range(1):
    trainStart = i * numOfPredicting
    trainEnd = numOfTraining + i * numOfPredicting
    predictEnd = numOfTraining + (i + 1) * numOfPredicting
    RandomList = label_pack[NAME].copy()['Time (sec)'].loc[trainStart:trainEnd].drop_duplicates().index
    RandomList = RandomList + trainStart
    predictList = np.arange(trainEnd, predictEnd,5)
    SVMDict,v,featureSetDict = train_model(df_pack,RandomList,NAME,TYPE,SIZEPRICE,levels,'FULL',isSpread)
    buySignalList = [False for i in range(len(predictList))]
    sellSignalList = [False for i in range(len(predictList))]
    pAndLLog = [0 for i in range(len(predictList))]
    
    for j in range(len(predictList) - 1):
        buy = False
        sell = False
        testIndexList =label_pack[NAME].copy()['Time (sec)'].\
            loc[predictList[j]:predictList[j+1]].drop_duplicates().index
        testIndexList = testIndexList + predictList[j]  # j* 5
        # constuct vpack again
        Basic_set = dict([(name,
                dict([(time,
                    dict([(ty,
                        dict([(SP,
                            dict([(str(le),
                    df_pack[name]['.'.join([ty,SP,str(le)])].iloc[idx])
                            for le in range(1,levels+1)]))
                                for SP in SIZEPRICE]))
                                    for ty in TYPE]))
                                        for idx, time in enumerate(df_pack[name]['Time (sec)']) if idx in testIndexList]))
                                            for name in list(df_pack)])

        v1_test = construct_basic_set(df_pack,Basic_set,testIndexList,NAME,TYPE,SIZEPRICE,levels)
        v2_test,v3_test,v4_test,v5_test = construct_time_insen_set(df_pack,Basic_set,testIndexList,NAME,TYPE,SIZEPRICE,levels)
        v6_test,v7_test,v8_test,v9_test,errorLog = construct_time_sensi_set(df_pack,testIndexList,NAME,msg_pack)
        # feature set option
        if feaType == 'BF':
            vPack = {'v1':v1_test}
        elif feaType == 'BTIS':
            vPack = {'v1':v1_test,'v2':v2_test,'v3':v3_test,'v4':v4_test,'v5':v5_test}
        elif feaType == 'BTS':
            vPack = {'v1':v1_test,'v6':v6_test,'v7':v7_test,'v8':v8_test,'v9':v9_test}
        else:
            vPack = {'v1':v1_test,'v2':v2_test,'v3':v3_test,'v4':v4_test,'v5':v5_test,'v6':v6_test,'v7':v7_test,\
                            'v8':v8_test,'v9':v9_test}
        for direct,svmModel in SVMDict.items():
            label_df_slice = slice_label_pack(testIndexList,NAME,label_pack,isSpread)
            testFeaturesAndLabels = assign_label_to_attributes(vPack,label_df_slice,True)
            testFeaturesAndLabels.loc['Label'] = label_df_slice.copy()
            # upward if predict is 1
            testFeaturesAndLabels.loc['Label.U'] = label_df_slice.mask(label_df_slice<= 0, -1).copy()
            # downward if predict is -1
            testFeaturesAndLabels.loc['Label.D'] = label_df_slice.mask(label_df_slice>= 0, 1).copy()
            # stationry if predict is -1
            testFeaturesAndLabels.loc['Label.S'] = label_df_slice.mask(label_df_slice!= 0, 1).mask(label_df_slice== 0, -1).copy()
            test_features_une = testFeaturesAndLabels.copy().replace(np.nan, 0).T
            #X_test = test_features_une[featureSet].drop([f'Label.{direct}'],axis = 1)
            X_test = test_features_une[featureSetDict[direct]]
            Y_test = svmModel.predict(X_test)
            #print(f'direction {direct} prediction is done.')
            if Y_test.any() > 0 and direct == 'U':
                buy = True
                buySignalList[j] = buy
                profit = test_features_une.iloc[-1]['BID_Size_1'] - test_features_une.iloc[0]['ASK_Size_1']
            if Y_test.any() < 0 and direct == 'D':
                sell = True
                sellSignalList[j] = sell
                profit = test_features_une.iloc[0]['BID_Size_1'] - test_features_une.loc[-1]['ASK_Size_1']
            if Y_test.any() < 0 and direct == 'S':
                sell = False
                buy = False
                sellSignalList[j] = sell
                buySignalList[j] = buy
                profit = 0
            pAndLLog[j] = profit
    tradingResultDf = pd.DataFrame({'Time Elapse':np.arange(0,numOfPredicting,5),'Buy': buySignalList,'Sell': sellSignalList,'P&L':pAndLLog})
    pAndLdfs[i] = tradingResultDf
#    return tradingResultDf

#%%
# plot 10,000 events backtesting p&L
fig = plt.figure()
fig.set_size_inches(18.5, 10.5)
ax1 = fig.add_subplot(211)
tradingResultDf['P&L'].plot(ax=ax1)
ax1.set_ylabel('P&L (USD)')
plt.hlines(y=0,xmin = 0,xmax = 2000,color='b', linestyle='--')
ax2 = fig.add_subplot(212)
ax2.set_ylabel('Cumulative P&L (USD)')
tradingResultDf['P&L'].cumsum().plot(ax=ax2)
plt.hlines(y=0,xmin = 0,xmax = 2000,color='b', linestyle='--')
plt.title('AAPL SVM trading on spread crossing using 10,000 events \n(Since openning of 6/21/2012)')
plt.savefig('10k_combined_spread.png', bbox_inches='tight', dpi=400)

# plot 30,000 events backtesting p&l
fig2 = plt.figure()
fig2.set_size_inches(18.5, 10.5)
ax3 = fig2.add_subplot(211)
res['P&L'].plot(ax=ax3)
ax1.set_ylabel('P&L (USD)')

ax4 = fig2.add_subplot(212)
ax4.set_ylabel('Cumulative P&L (USD)')
res['P&L'].cumsum().plot(ax=ax4)
xposition = [2000, 4000]
for xc in xposition:
    plt.axvline(x=xc, color='k', linestyle='--')
plt.hlines(y=0,xmin = 0,xmax = 6000,color='b', linestyle='--')
plt.title('AAPL SVM trading on mid-price using 30,000 events \n(Since beginning of 6/21/2012)')
plt.savefig('30k_combined.png', bbox_inches='tight', dpi=400)

# ax2.plot(buys.index, results.short_mavg.ix[buys.index],
#                  '^', markersize=10, color='m')
# ax2.plot(sells.index, results.short_mavg.ix[sells.index],
#                  'v', markersize=10, color='k')
# plt.legend(loc=0)

