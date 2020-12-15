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
order_names_dict =dict(AMZN = 'AMZN_orderbook_10.csv',AAPL = 'AAPL_orderbook_10.csv',GOOG = 'GOOG_orderbook_10.csv',
                    INTC='INTC_orderbook_10.csv',MSFT = 'MSFT_orderbook_10.csv')
msg_names_dict =dict(AMZN = 'AMZN_message_10.csv',AAPL = 'AAPL_message_10.csv',GOOG = 'GOOG_message_10.csv',
                    INTC='INTC_message_10.csv',MSFT = 'MSFT_message_10.csv')

# %%
# preparation, read in data
name_list = ['AMZN','AAPL','GOOG','INTC','MSFT']

AMZN_msg_df = pd.read_csv(msg_names_dict['AMZN'],names = ['Time (sec)',
    'Event Type',
    'Order ID',
    'Size',
    'Price',
    'Direction'])
AMZN_odr_df = pd.read_csv(order_names_dict['AMZN'],
                    names = [f'{ty}.{VP}.{i}' 
                        for i in range(1,11) for ty in ['ASK','BID'] 
                            for VP in ['Price','Size'] ])
AMZN_odr_df['Time (sec)'] = AMZN_msg_df['Time (sec)']

# %%
#df_pack = {'AMZN':AMZN_odr_df,'AAPL': AAPL_odr_df, 'GOOG': GOOG_odr_df, 'INTC':INTC_odr_df , 'MSFT': MSFT_odr_df}
df_pack = {NAME: AMZN_odr_df}
msg_pack = {NAME:AMZN_msg_df}
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
# more accurate calculation of spread movement
# for idx, row in label_df.iterrows():
#     time = label_df.at[idx,'Time (sec)']
#     closestTime1 = min(timeLog, key=lambda x:abs(x-(time-1)))
#     closestTime1Idx = label_df.index[label_df['Time (sec)']==closestTime1].to_list()[0]
#     if df_pack[name].at[idx]
label_df['Spread.crossing'] = label_df.apply(lambda x: check_spread_crossing_direction(x['Spread.BID.Diff'],x['Spread.ASK.Diff']),axis = 1)
label_df = label_df.fillna(0)
label_pack = {NAME:label_df}


# %%
# label set and dfpack has all the rows, ie, 40093
# basic set whole  has unique rows



# %%
# this RandomList is the index(int)
targetList = label_pack[NAME][targetEstimator]
proportion = targetList.value_counts()/len(targetList)
RandomList = []
for i in [0, 1, -1]:
    RandomList += list(targetList[targetList == i].sample(int(1500*proportion[i])).index)
rest = 1500 - len(RandomList)
for r in range(rest):
    temp = list(targetList[targetList == 0].sample(1).index)[0]
    if temp not in RandomList:
        RandomList += [temp]

# %%
# sampling 1500 data points randomly
#randomList = random.sample(range(1, len(df_pack['AMZN'])), 1500)
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


# %%
# construct sets
# v1 should indexed by time
def construct_basic_set(df_pack,randomList,name,Types,SizePrice,levels):
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


# %%
v1 = construct_basic_set(df_pack,RandomList,NAME,TYPE,SIZEPRICE,levels)


# %%
def construct_time_insen_set(df_pack,randomList,name,Type,SizePrice,levels):
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


# %%
v2,v3,v4,v5 = construct_time_insen_set(df_pack,RandomList,NAME,TYPE,SIZEPRICE,levels)


# %%
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
    
    


# %%
v6,v7,v8,v9,errorLog = construct_time_sensi_set(df_pack,RandomList,NAME,msg_pack)


# %%

#targetEstimator = 'Spread.crossing'
#feaType = 'BTS'
# %%
feature_set_list = ['BF','BTIS','BTS','ALLF']
if feaType == 'BF':
    vPack = {'v1':v1}
elif feaType == 'BTIS':
    vPack = {'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5}
elif feaType == 'BTS':
    vPack = {'v1':v1,'v6':v6,'v7':v7,'v8':v8,'v9':v9}
else:
    vPack = {'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5,'v6':v6,'v7':v7,'v8':v8,'v9':v9}


# %%
def assign_label_to_attributes(vPack,labelSet):
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
    dfList = []
    for name,sets in vPack.items():
        dfList.append(pd.DataFrame.from_dict(sets))
    attributesAndLabelsDf = pd.concat(dfList)
    return attributesAndLabelsDf


# %%
def slice_label_pack(randomList,name,label_pack, metric):
    '''
    metic: Spread, MidPrice
    return a pd.Series sorted by time
    '''
    sortedList = sorted(randomList)
    if metric =='Spread':
        temp = list(label_pack[name].iloc[sortedList]['Spread.crossing'])
        return pd.Series(temp,index = label_pack[name].iloc[sortedList]['Time (sec)'])
    elif metric == 'MidPrice':
        temp = list(label_pack[name].iloc[sortedList]['MID.Dir'])
        return pd.Series(temp,index = label_pack[name].iloc[sortedList]['Time (sec)'])
    else:
        raise ValueError


# %%
# 82/1500 spread > 0
# 74 midprice < 0 72 midprice >0 and 1354 are equal to 0.
# make sure you run only once
# here to choose you use Spread or MidPrice as indicator.
label_df_slice = slice_label_pack(RandomList,NAME,label_pack,'Spread')


# %%

#label_df_slice.value_counts()


# %%
# featuresAndLabels has dim 44,1500
featuresAndLabels = assign_label_to_attributes(vPack,label_df_slice)
featuresAndLabels.loc['Label'] = label_df_slice.copy()
featuresAndLabels.loc['Label.U'] = label_df_slice.mask(label_df_slice<= 0, -1).copy()
featuresAndLabels.loc['Label.D'] = label_df_slice.mask(label_df_slice>= 0, 1).copy()
featuresAndLabels.loc['Label.S'] = label_df_slice.mask(label_df_slice!= 0, 1).mask(label_df_slice== 0, -1).copy()



# %%
#featuresAndLabels.tail()
#featuresAndLabels.shape



# %%
def group_attributes(c):
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
featuresDict,featuresList = group_attributes(vPack)

# %%
# the following are proformed based on Sklearn package
movementDir = ['U','D','S'] #classes=["U", "D", "S"]
lb = LabelBinarizer()
test_features = featuresAndLabels.copy().replace(np.nan, 0).T
# test_features has dim 1500,44
for label in movementDir:
    X_train = test_features.drop([f'Label.{label}'],axis = 1)
    Y_train_bin = lb.fit_transform(test_features[f'Label.{label}'])
    Y_train = test_features[f'Label.{label}']
    svc = OneVsRestClassifier(SVC(C = 0.25, kernel="poly",degree=2,decision_function_shape='ovr'))
    print(f'For Label {label} in testing indicator {targetEstimator} on feature set {feaType}')
    #print(cross_validate(svc, X_train, Y_train_bin, cv=StratifiedKFold(10), scoring=["recall"],verbose=1))
    #fitTime, predictTime, precision, recall, F1 = 
    print(cross_validate(svc, X_train, Y_train, cv=StratifiedKFold(10), scoring=["precision",'recall','f1'],verbose=0))
    #print(f'The fitting time for Label {label} is {fitTime}')



# %%
# this is to verify if date is the same
b = list(label_df_slice.index)
a = list(featuresAndLabels.columns)
l = []
for ii in b:
    if ii not in a:
        l+= [ii]


# %%
svc = SVC(C = 0.25, kernel="poly",degree=2,decision_function_shape='ovr')
svcDict = dict()
#rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),scoring='accuracy',verbose =1)
#rfecv.fit(test_features.drop(['Label'],axis = 1).to_numpy(), np.array(XtrainTransformed))
svc_ovr = SVC(C = 0.25, kernel="poly",degree=2,decision_function_shape='ovr')


# %%
#classification_report


# %%
# stratified k fold statistics: above 90%
cross_val_score(svc, X_train, Y_train, cv=StratifiedKFold(10), scoring="f1_micro",verbose=1)


# %%
# try cross_validate
cross_validate(svc, X_train, Y_train, cv=StratifiedKFold(10), scoring=["accuracy",'f1_micro','neg_mean_squared_error'],verbose=1)


# %%
# this is to verify the work by cross_val_score 
# test_size = 0.1 since we are doing 1o folds, so 1350/1500
# this is a multilabel classifier, so should use 'average' as micro.
Y_train_bin = lb.fit_transform(test_features[f'Label'])
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
numSplits = sss.get_n_splits(X_train,Y_train_bin)
precisionList = dict()
recallList = dict()
APList = dict()
classifyReports = dict()
preRecFSupports = dict()
for j, (train_index, test_index) in enumerate(sss.split(X_train,Y_train_bin)):
    X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train_fold, y_test_fold = Y_train_bin[train_index], Y_train_bin[test_index]
    svc.fit(X_train_fold,y_train_fold)
    for num in range(3):
        print(f'Number of support vectors in the {num} class is {svc.estimators_[num].n_support_}')
    y_predict = svc.predict(X_test_fold)
    y_score = svc.decision_function(X_test_fold)
    #print(y_test_fold)
    #average precision score do not support multiple class.
    #average_precision = average_precision_score(y_test_fold, y_score)
    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    # we can get U,D,S information from here.
    for i in range(3):
        precision[i], recall[i], _ = precision_recall_curve(y_test_fold[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(y_test_fold[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_fold.ravel(),y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test_fold, y_score,
                                                    average="micro")
    precisionList[j] = precision
    recallList[j] = recall
    APList[j] = average_precision
    # create classification reports for each of the subfold
    report = classification_report(y_test_fold, y_predict)
    prfs = precision_recall_fscore_support(y_test_fold, y_predict,1,average="micro")
    classifyReports[j] = report
    preRecFSupports[j] = prfs
microAverageList = [v["micro"] for v in APList.values()]
microAverage = np.mean(microAverageList)
print('Average precision score, micro-averaged over all Spread Crossing classes: {0:0.2f}'
        .format(microAverage))


# %%
print(classifyReports[2])


# %%
print(preRecFSupports[4])


# %%
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Average precision score\n micro-averaged over all Spread Crossing classes: AP={0:0.2f}'
    .format(average_precision["micro"]))


# %%
svc.decision_function


# %%
from itertools import cycle
# setup plot details
n_classes = 3
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

plt.figure(figsize=(7, 8))
# an iso-F1 curve contains all points in the precision/recall space whose F1 scores are the same. 
f_scores = np.linspace(0.2, 0.95, num=5)
lines = []
labels = []
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
lines.append(l)
labels.append('Micro-average Precision-recall (area = {0:0.2f})'
              ''.format(average_precision["micro"]))

for i,j, color in zip(range(n_classes),['Upward','Downward','Stationary'], colors):
    l, = plt.plot(recall[i], precision[i], color=color, lw=2)
    lines.append(l)
    labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                  ''.format(j, average_precision[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve for Spread Crossing')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


# %%
n_support_


# %%
#Fit label binarizer and transform multi-class labels to binary labels.
YtrainTransformed = lb.fit_transform(Y_train.astype(np.float64))


# %%
# this does not work
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10),scoring='accuracy',verbose =1)
rfecv.fit(test_features.drop(['Label'],axis = 1).to_numpy(), test_features['Label'])


# %%



# %%



# %%
def assemble_attri_label_df(attributesDict,label_slice):
    attributesDf = pd.DataFrame(attributesDict)


# %%
#not used
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
    featureImportance = []
    for split_attribute_name in featureList:
        featureImportance.append(InfoGain(data,split_attribute_name,target_name))
    featureImportance.sort(reverse = True)
    weights = 0
    for idx, weight in enumerate(featureImportance):
        weights += weight
        if weights > 0.95: 
            return featureImportance[:idx+1]


# %%
# not used, potentially useful
res = select_attributes_based_on_IG(featuresAndLabels,"Label",featuresList)

# %%
# check the closest time and row count
closestTime1 = min(timeLog, key=lambda x:abs(x-(34202.77893	+ 1)))
ind = label_df.index[label_df['Time (sec)']==closestTime1].to_list()[0]
print(f'ind: {ind},timeClose: {closestTime1}')


# %%
len(label_df_slice)


# %%



# %%



# %%


