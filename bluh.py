from hmmlearn import hmm
from scipy.stats import norm
from numpy.random import normal,uniform
from pandas.plotting import register_matplotlib_converters
import numpy
import numpy as np
import pandas as pd
import sys

def build_model(data,states=10,L=20):
    model = hmm.GMMHMM(n_components=states, covariance_type="diag")
    T=int(len(data)/L)
    model.fit(data, [ T for l in range(0,L)])
    return model

def predict_next_value(obs,model,states=10):
    n_x=len(obs) #number of observations
    n_f=len(obs[0])
    if model.startprob_.sum()!=1:
        model.startprob_=np.full(shape=states,fill_value=1/states)
    for s in range(states):
        if model.transmat_[s].sum() != 1:
            model.transmat_[s]=np.full(shape=states,fill_value=1/states)
        model.transmat_[s]=model.transmat_[s] /model.transmat_[s].sum()
    mtx_p=model.predict_proba(obs) # n_x * states
    ret=[]
    for x in range(0,n_x):
        feet=np.zeros(shape=n_f)
        for s in range(states):
            prb=mtx_p[x][s] # the probability that sample x is in state s
            for y in range(states):
                prb*=model.transmat_[s][y] # probability that state s goes to state y
                #print(model.means_[y])
                for f in range(n_f):
                    feet[f]+= prb * model.means_[y][0][f]
        ret.append(feet)
    return ret


    return next_value

def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, exposure, equity, settings):
    ''' This system uses trend following techniques to allocate capital into the desired equities'''

    nMarkets=CLOSE.shape[1]
    #print(equity)
    #print(DATE[0])
    #print(CLOSE[-50:,][49]) #closing prices from periodshorter days ago
    #print(CLOSE[-100:,][99])

    period=settings['lookback']
    states=settings['states']

    df=pd.DataFrame(CLOSE).pct_change().tail(period-1) *10000
    today=df.tail(1)
    #print(settings['guess'],today[0].to_numpy())
    past_data=df.head(period-2)
    settings['error']+=abs(settings['guess']-today[0].to_numpy()[0])
    #print(type(DATE[0]))
    p='avg abs error {} states {} lookback {}'.format(settings['error']/period,states,period)
    model=build_model(past_data.to_numpy(),states=states,L=5)
    settings['guess']=1
    pred=predict_next_value(today.to_numpy(),model,states=states)[0]
    #print(pred)
    del model
    weights = pred

    return weights, settings


def mySettings():
    ''' Define your trading system settings here '''

    settings= {}

    # S&P 100 stocks
    # settings['markets']=['CASH','AAPL','ABBV','ABT','ACN','AEP','AIG','ALL',
    # 'AMGN','AMZN','APA','APC','AXP','BA','BAC','BAX','BK','BMY','BRKB','C',
    # 'CAT','CL','CMCSA','COF','COP','COST','CSCO','CVS','CVX','DD','DIS','DOW',
    # 'DVN','EBAY','EMC','EMR','EXC','F','FB','FCX','FDX','FOXA','GD','GE',
    # 'GILD','GM','GOOGL','GS','HAL','HD','HON','HPQ','IBM','INTC','JNJ','JPM',
    # 'KO','LLY','LMT','LOW','MA','MCD','MDLZ','MDT','MET','MMM','MO','MON',
    # 'MRK','MS','MSFT','NKE','NOV','NSC','ORCL','OXY','PEP','PFE','PG','PM',
    # 'QCOM','RTN','SBUX','SLB','SO','SPG','T','TGT','TWX','TXN','UNH','UNP',
    # 'UPS','USB','UTX','V','VZ','WAG','WFC','WMT','XOM']

    # Futures Contracts
    settings['markets']  = ['F_GC']

    '''settings['markets']  = ['CASH','F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD',
    'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC','F_FV', 'F_GC',
    'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP',
    'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU',
    'F_S','F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US','F_W', 'F_XX',
    'F_YM']'''
    settings['beginInSample'] = '20130506'
    settings['endInSample'] = '20150606'
    settings['lookback']= 400
    settings['budget']= 10**6
    settings['slippage']= 0
    settings['states']=4
    settings['guess']=1
    settings['error']=0

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    register_matplotlib_converters()
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
    #sys.exit(0)
