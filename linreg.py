import quantopian.algorithm as algo
from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from quantopian.pipeline.factors import (
    SimpleMovingAverage,
    CustomFactor,
    Returns,
    MACDSignal,
    BollingerBands
)
from quantopian.pipeline.data.quandl import cboe_vix as vix
from sklearn import ensemble, preprocessing, metrics, linear_model
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from quantopian.pipeline import CustomFilter

WINDOW_LENGTH=20

def initialize(context):
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=3),
    )
    algo.attach_pipeline(make_pipeline(), 'pipeline')
    context.df=None


def make_pipeline():
    pipe = Pipeline(
        screen=specific_assets_filter(symbols('GLD','SPY'))
    )
    bb_gld = BollingerBands(window_length=WINDOW_LENGTH, k=2,mask=specific_assets_filter(
            symbols('GLD')
    ))
    bb_spy = BollingerBands(window_length=WINDOW_LENGTH, k=2,mask=specific_assets_filter(
            symbols('SPY')
    ))
    pipe.add(VIXFactor(mask=specific_assets_filter(
            symbols('GLD')
        )),'vix')
    pipe.add(bb_gld.lower, 'bb_gld_lower')
    pipe.add(bb_gld.middle, 'bb_gld_middle')
    pipe.add(bb_gld.upper, 'bb_gld_upper')
    pipe.add(bb_spy.lower, 'bb_spy_lower')
    pipe.add(bb_spy.middle, 'bb_spy_middle')
    pipe.add(bb_spy.upper, 'bb_spy_upper')
    return pipe

def before_trading_start(context, data):
    build_df(context,data)
    context.model=build_model(context,data)
    #build_model(context,data)
    """
    Called every day before market open.
    """
    #context.output = algo.pipeline_output('pipeline')

def build_df(context,data):
    output=pipeline_output('pipeline')
    outdf=pd.DataFrame(columns=output.columns)
    d={}
    for c in outdf.columns:
        d[c]=output[c].max()
    d["gld_price"]=data.current(symbol('GLD'), 'close')
    outdf=outdf.append(d,ignore_index=True)
    if context.df is None:
        context.df=outdf
    else:
        context.df=context.df.append(outdf)
        if(len(context.df)>40):
            context.df=context.df.tail(40)
        
def build_model(context,data):
    y=context.df['gld_price']
    x=context.df.drop(['gld_price'],axis=1)
    model=LinearRegression()
    model.fit(x,y)
    return(model)


def rebalance(context, data):
    #this is dogshit
    if buy(context,data) == True:
        order_target_percent(symbol('GLD'), -1.1)
    else:
        order_target_percent(symbol('GLD'), 1.1)


def buy(context,data):
    output=pipeline_output('pipeline')
    outdf=pd.DataFrame(columns=output.columns)
    d={}
    for c in outdf.columns:
        d[c]=output[c].max()
    outdf=outdf.append(d,ignore_index=True)
    pred=context.model.predict(outdf)
    return( pred[0] > data.current(symbol('GLD'), 'price'))
    

class VIXFactor(CustomFactor):  
    window_length = WINDOW_LENGTH 
    inputs = [vix.vix_close]
    def compute(self, today, assets, out, vix):
        # https://www.quantopian.com/posts/upcoming-changes-to-quandl-datasets-in-pipeline-vix-vxv-etc-dot
        out[:] = vix[-1]

class Average_True_Range(CustomFactor):
    #https://www.quantopian.com/posts/custom-factor-atr
    window_length = WINDOW_LENGTH
    def compute(self, today, assets, out, close, high, low):
        lb = self.window_length
        atr = np.zeros(len(assets), dtype=np.float64)
        a=np.array(([high[1:(lb)]-low[1:(lb)],
                    abs(high[1:(lb)]-close[0:(lb-1)]),
                    abs(low[1:(lb)]-close[0:(lb-1)])]))
        b=a.T.max(axis=2)
        c=b.sum(axis=1)
        atr=c /(lb-1)
        out[:] = atr
        
def specific_assets_filter(assets):
    #https://www.quantopian.com/posts/custom-filter-for-stocks-in-a-list
    sids = set(map(int, assets))  
    is_my_sid = np.vectorize(lambda sid: sid in sids)

    class SpecificAssets(CustomFilter):  
        inputs = ()  
        window_length = 1  
        def compute(self, today, assets, out):  
            out[:] = is_my_sid(assets)

    return SpecificAssets()