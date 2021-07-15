import pandas as pd
import numpy as np
from statsmodels.tsa.filters.filtertools import recursive_filter
import math 
from datetime import date,datetime
import logging
from tigerml.core.reports import create_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bokeh.models.formatters import DatetimeTickFormatter


#imputes missing values
def impute_missing_values(data):
    logging.info('Entering impute_missing_vlaues function')
    m_df=pd.DataFrame(data.isna().sum())
    m_df=m_df.reset_index()
    m_df.columns=['column_name','count']
    m_df=m_df.loc[m_df['count']>0,:]
    for c in m_df['column_name'].tolist():
        data[c]=data[c].fillna(method='bfill')
        if data[c].isna().sum()>0:
            data[c]=data[c].fillna(method='ffill')
    logging.info('Leaving impute_missing_vlaues function')
    return data


#calculates the decay rate of the halflife
def adstock_decay(x):
    return math.exp(math.log(0.5)/math.log(x))


#returns best adtsock or stocking lag of the variable based on correlation with dv 
def get_best_transformation(corr,dv):
    corr=corr.loc[corr['index']!=dv,:]
    corr=corr.loc[corr[dv]==corr[dv].max(),:]
    return corr.iloc[0,:]['index']

    
#adtsock calculation method
def get_adstock(col,halflife):
    ad=adstock_decay(halflife)
    ad_stock=recursive_filter(col,ad)
    return ad_stock

#performs adtsock analysis based on config
def adstock_analysis(data,feature_config,dv,global_config):
    logging.info('Entering adtsocks analysis function')
    adstock_cols=list(feature_config['adstocks'].keys())
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    output_df=pd.DataFrame()
    for a in adstock_cols:
        adstock_df=pd.DataFrame()
        adstock_df[a]=data[a]
        hl=feature_config['adstocks'][a]
        hl=hl.split(',')
        hl=[int(x) for x in hl]
        adstock_df[dv]=data[dv]
        for h in hl:
            c=f'{a}_h{h}'
            adstock=get_adstock(data[a],h)
            adstock_df[c]=adstock
        corr=adstock_df.corr().reset_index()
        corr=corr.loc[:,['index',dv]]
        dummy_df=pd.DataFrame()
        columns=['column']
        dummy_df['column']=[a]
        for h in hl:
            columns.append(f'h{h}')
        for c in columns:
            if c !='column':
                dummy_df[c]=corr.loc[corr['index']==f'{a}_{c}',:][dv].tolist()
        output_df=output_df.append(dummy_df)
        if feature_config['select_best_transformation']:
            fn=get_best_transformation(corr,dv)
            data[fn]=adstock_df[fn]
        else:
            data=pd.concat([data,adstock_df.drop([dv,a],axis=1)],axis=1)
    output_df['transformation']='adstock'
    logging.info('Leaving adtsocks analysis function')
    return data,output_df

#returns stocking lag of the variable takes lag as input
def stocking_lag(col,lag):
    ads_decay=1
    adstock_decays=[1 for i in range(1,lag)]
    adf=pd.DataFrame()
    adf['x']=col
    for i in range(len(adstock_decays)):
        adf[f'lag_{i+1}']=adf['x'].shift(periods=i+1)
        adf[f'lag_{i+1}']=adf[f'lag_{i+1}']*adstock_decays[i]
        adf[f'lag_{i+1}']=adf[f'lag_{i+1}'].fillna(0)
    adf['adstock']=adf['x']
    for i in range(len(adstock_decays)):
        adf['adstock']=adf['adstock']+adf[f'lag_{i+1}']
    return adf['adstock']


#performs adtsock analysis based on config
def s_lags_analysis(data,feature_config,dv,global_config):
    logging.info('Entering stocking lags analysis function')
    today = date.today()
    d1 = today.strftime("%d-%m-%Y")
    sl_cols=list(feature_config['stocking_lags'].keys())
    output_df=pd.DataFrame()
    for a in sl_cols:
        sl_df=pd.DataFrame()
        sl_df[a]=data[a]
        lags=feature_config['stocking_lags'][a]
        lags=lags.split(',')
        lags=[int(x) for x in lags]
        sl_df[dv]=data[dv]
        for l in lags:
            c=f'{a}_sl{l}'
            sl=stocking_lag(data[a],l)
            sl_df[c]=sl
        corr=sl_df.corr().reset_index()
        corr=corr.loc[:,['index',dv]]
        dummy_df=pd.DataFrame()
        columns=['column']
        dummy_df['column']=[a]
        for h in lags:
            columns.append(f'sl{h}')
        for c in columns:
            if c!='column':
                dummy_df[c]=corr.loc[corr['index']==f'{a}_{c}',:][dv].tolist()
        output_df=output_df.append(dummy_df)
        if feature_config['select_best_transformation']:
            fn=get_best_transformation(corr,dv)
            data[fn]=sl_df[fn]
        else:
            data=pd.concat([data,sl_df.drop([dv,a],axis=1)],axis=1)
    output_df['transformation']='stocking_lag'
    logging.info('Leaving stocking lags analysis function')
    return data,output_df


def tranformations_bivaraite_plots(data,feature_config,global_config):
    d2=datetime.now()
    d2=d2.strftime("%m%d%Y_%H%M%S")
    cols=list(feature_config['adstocks'].keys())+list(feature_config['stocking_lags'].keys())
    cols=list(set(cols))
    feature_dict={}
    for c in cols:
        feature_dict[c]={}
        if c in list(feature_config['adstocks'].keys()):
            a=feature_config['adstocks'][c].split(',')
            feature_dict[c]['adstocks']=[int (x) for x in a]
        if c in list(feature_config['stocking_lags'].keys()):
            a=feature_config['stocking_lags'][c].split(',')
            feature_dict[c]['stocking_lags']=[int (x) for x in a]
    report_dict={'Transformations Bivariate Plots':{}}
    for c in cols:
        t_df=pd.DataFrame()
        t_df[global_config['time_column']]=data[global_config['time_column']].astype('datetime64[ns]')
        t_df[c]=data[c]
        t_df[global_config['dv']]=data[global_config['dv']]
        fd_keys=feature_dict[c].keys()
        for f in list(fd_keys):
            if f=='adstocks':
                halflifes=feature_dict[c]['adstocks']
                for h in halflifes:
                    t_df[f'halflife_{h}']=get_adstock(data[c],h)
            if f =='stocking_lags':
                sls=feature_dict[c]['stocking_lags']
                for s in sls:
                    t_df[f'stocking_lag_{s}']=stocking_lag(data[c],s)
        ycols=[x for x in t_df.columns.tolist() if x!=global_config['time_column']]
        #fig = make_subplots(specs=[[{"secondary_y": True}]])
        tc=global_config['time_column']
        dv=global_config['dv']
        t_df=t_df.set_index(tc)
        plot=t_df.hvplot.line(x=tc, y=ycols, legend='top', height=600, width=1000, by=['index.year','index.month']).opts(legend_position='top',xrotation=90,xformatter = DatetimeTickFormatter(months = '%b %Y'))

        cor=t_df.corr().reset_index()
        cor=cor.loc[cor['index']!=global_config['dv'],['index',global_config['dv']]]
        cor.columns=['tranformation','correlation']
        cor=cor.reset_index(drop=True)
        report_dict['Transformations Bivariate Plots'][c]={'plot':plot,'correlation':cor}
        print(report_dict)
    create_report(report_dict, name=f'tranformation_bivariate_plots_{global_config["run_name"]}_{d2}', path='Output/feature_transformations/', format='.html', split_sheets=True, tiger_template=False)
    
    


# function which calls all above functions
def feature_eng(data,global_config,feature_config):
    d2=datetime.now()
    d2=d2.strftime("%m%d%Y_%H%M%S")
    try:
        data=impute_missing_values(data)
    except Exception as e:
        logging.error("Exception occurred while imputing missing values", exc_info=True)
    try:
        if len(feature_config['adstocks'].keys())>0:
            data,ads_df=adstock_analysis(data,feature_config,global_config['dv'],global_config)
    except Exception as e:
        logging.error("Exception occurred while doing adtsock analysis", exc_info=True)
    try:
        if len(feature_config['stocking_lags'].keys())>0:
            data,sl_df=s_lags_analysis(data,feature_config,global_config['dv'],global_config)
    except Exception as e:
        logging.error("Exception occurred while doing stocking lags analysis", exc_info=True)
    try:
        ads_df.columns=[x.replace('h','halflife_') if len(x)==2 else x for x in ads_df.columns.tolist()]
        sl_df.columns=[x.replace('sl','stocking_lag_') if len(x)==3 else x for x in sl_df.columns.tolist()]
        final_df=ads_df.merge(sl_df,on='column',how='outer')
        final_df['transformation']=np.where(final_df['transformation_x'].isnull(),final_df['transformation_y'],final_df['transformation_x'])
        final_df.drop(['transformation_x','transformation_y'],axis=1,inplace=True)
        cl=final_df.columns.tolist()
        clss=['column','transformation']+[x for x in cl if x not in ['column','transformation']]
        final_df=final_df.loc[:,clss]
        final_df.to_csv(f'Output/feature_transformations/transformations_{global_config["run_name"]}_{d2}.csv',index=False)
    except Exception as e:
        logging.error("Exception occurred while saving adtsock and stockling lag analysis", exc_info=True)
    
    try:
        tranformations_bivaraite_plots(data,feature_config,global_config)
    except Exception as e:
        logging.error("Exception occurred while getting transformations time series plots", exc_info=True)
    
    return data

