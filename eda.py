import pandas as pd
import numpy as np
from statsmodels.tsa.filters.filtertools import recursive_filter
import math 
from datetime import date,datetime
from tigerml.eda import EDAReport
import logging
import hvplot.pandas
from hvplot import hvPlot
from tigerml.core.reports import create_report
import warnings
from bokeh.models import *
import holoviews as hv
hv.extension('bokeh')
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from bokeh.models.formatters import DatetimeTickFormatter

warnings.filterwarnings("ignore")

#gets max date for certain column
def max_date(x,time_column,data):
    d=data.loc[:,['date_sunday',x]]
    d=d.dropna()
    return d['date_sunday'].max()

#gets min date for certain column
def min_date(x,time_column,data):
    d=data.loc[:,['date_sunday',x]]
    d=d.dropna()
    return d['date_sunday'].min()

#gets the count of zeros in a particular column
def zero_counts(x,data):
    d=data.loc[data[x]==0,:]
    return d.shape[0]

#gets us all the univariate netrics for all the variables of a column
def univariate_analysis(data,time_column):
    logging.info('Entering Univariate function')
    numeric_cols= data.select_dtypes([np.number]).columns.tolist()
    dtypes=data[numeric_cols].describe().T.reset_index()
    dtypes.rename(columns={'index':'column_name'},inplace=True)
    dtypes['10%']=dtypes['column_name'].apply(lambda x:data[x].quantile(0.1))
    dtypes['90%']=dtypes['column_name'].apply(lambda x:data[x].quantile(0.9))
    dtypes['Missing_count']=dtypes['column_name'].apply(lambda x:data[x].isna().sum())
    dtypes['missing_per']=dtypes['Missing_count']/data.shape[0]
    dtypes.rename(columns={'count':'nRows_in_DataSet','25%':'P25','50%':'Median','75%':'P75','10%':'P10','90%':'P90'},inplace=True)
    dtypes['max_date']=dtypes['column_name'].apply(lambda x:max_date(x,time_column,data))
    dtypes['min_date']=dtypes['column_name'].apply(lambda x:min_date(x,time_column,data))
    dtypes['zero_counts']=dtypes['column_name'].apply(lambda x:zero_counts(x,data))
    remaining_cols=[ x for x in data.columns.tolist() if (x not in numeric_cols) & (x!=time_column)]
    ctypes=pd.DataFrame()
    if len(remaining_cols)>0:
        ctypes['column_name']=remaining_cols
        ctypes['Missing_count']=ctypes['column_name'].apply(lambda x:data[x].isna().sum())
        ctypes['missing_per']=ctypes['Missing_count']/data.shape[0]
        ctypes['n_categories']=ctypes['column_name'].apply(lambda x:data[x].unique().shape[0])
        ctypes['mode']=ctypes['column_name'].apply(lambda x:data[x].mode().tolist())
        ctypes['max_date']=ctypes['column_name'].apply(lambda x:max_date(x,time_column,data))
        ctypes['min_date']=ctypes['column_name'].apply(lambda x:min_date(x,time_column,data))
        ctypes['zero_counts']=ctypes['column_name'].apply(lambda x:zero_counts(x,data))
    logging.info('coming out of univraite function')
    return dtypes,ctypes

#identifies outliers
def return_outlier_ids(data,time_column,col):
    data=data.loc[:,[time_column,col]]
    mean=data[col].mean()
    std=data[col].std()
    data['z']=(data[col]-mean)/std
    upperlimit=mean+3*std
    lowerlimit=mean-3*std
    data['o_i']=np.where((data['z']>3)|(data['z']<-3),'outlier','no')
    return data.loc[data['o_i']=='outlier',:]['date_sunday'].astype(str).tolist()        

#identifies outliers
def identify_outliers(data,time_column):
    logging.info('entering identify outliers function')
    numeric_cols= data.select_dtypes([np.number]).columns.tolist()
    o_df=pd.DataFrame()
    o_df['column_name']=numeric_cols
    o_df['outlier_ids']=o_df['column_name'].apply(lambda x:return_outlier_ids(data,time_column,x))
    logging.info('coming out of identify outliers function')
    return o_df

# correlation analysis
def correlation_analysis(data,global_config):
    logging.info('entering correlation analysis function')
    columns=data.columns.tolist()
    groups=[x.split('.')[0] for x in columns]
    groups=list(set(groups))
    cols_dict={g:[d for d in columns if g in d] for g in groups}
    # for g in groups:
    #     cols_dict[g]=[d for d in columns if g in d]
    sales_group=['lowespos']
    fb_group=['fbadman_fbads_p', 'rtop_fbpost_o']
    google_group=['GA_search_p']
    corp_sponsors=['corp_gum_social', 'corp_gum_broadcast','corp_mlb_social','corp_mlb_broadcast']
    comark=['lowes_comaek_p']
    external_group=['ext']
    email_group=['merkle_email_p']
    insta=['rtop_instastory_o','rtop_instapost_o']
    twitter=['rtop_twitterpost_o']
    lowes_prom=['lowes_prom']
    ad_groups=[fb_group,google_group,corp_sponsors,external_group,email_group,insta,twitter,comark,lowes_prom]
    ad_groups_n=['fb_group','google_group','corp_sponsors','external_group','email_group','insta','twitter','comark','lowes_prom']
    d1=datetime.now()
    d1=d1.strftime("%m%d%Y_%H%M%S")
    wb=f'Output/eda_output/correlation_analysis_{global_config["run_name"]}_{d1}.xlsx'
    writer=pd.ExcelWriter(wb)
    overall_corr=data.corr().reset_index()
    overall_corr=overall_corr.loc[:,['index',global_config['dv']]]
    overall_corr.to_excel(writer,'Correlation with DV',index=False)
    for i in range(0,len(ad_groups)):
        g=ad_groups[i]
        n=ad_groups_n[i]
        cols=[]
        for t in g:
            cols.extend(cols_dict[t])
        cols.extend(cols_dict['lowespos'])
        data_sub=data.loc[:,cols]
        corr=data_sub.corr().reset_index()
        corr.to_excel(writer,n,index=False)
    writer.close()
    logging.info('coming out of correlation analysis function')

#prepared eda report using tigerml
def eda_report_generation(data,global_config):
    d1=datetime.now()
    d1=d1.strftime("%m%d%Y_%H%M%S")
    data[global_config['time_column']]=data[global_config['time_column']].astype(str)
    an = EDAReport(data, y=global_config['dv'])
    an.get_report(quick=True,name=f'eda_report_{global_config["run_name"]}_{d1}',save_path='output/eda_output/')


#adds seasonality column
def get_seasonality_column(data,dv,level,time_column):
    if level=='week':
        data[time_column]=pd.to_datetime(data[time_column])
        data['week']=data[time_column].dt.week
        data_w=data.groupby('week',as_index=False).agg({dv:'mean'})
        avg=data[dv].mean()
        data_w[dv]=data_w[dv]/avg
        data_w.rename(columns={dv:'s_index_weekly'},inplace=True)
        data=data.merge(data_w,on='week',how='left')
        data.drop('week',axis=1,inplace=True)
        return data
    elif level=='month':
        data[time_column]=pd.to_datetime(data[time_column])
        data['week']=data[time_column].dt.month
        data_w=data.groupby('month',as_index=False).agg({dv:'mean'})
        avg=data[dv].mean()
        data_w[dv]=data_w[dv]/avg
        data_w.rename(columns={dv:'s_index_monthly'},inplace=True)
        data=data.merge(data_w,on='month',how='left')
        data.drop('month',axis=1,inplace=True)
        return data

#adds trend colmn
def add_trend_column(data):
    data= data.reset_index(drop=True)
    data['trend']=data.index+1
    return data



def apply_formatter(plot, element):
    p = plot.state
    # create secondary range and axis
    p.extra_y_ranges = {"twiny": Range1d(start=200000, end=1000000)}
    p.add_layout(LinearAxis(y_range_name="twiny"), 'left')
    # set glyph y_range_name to the one we've just created
    glyph = p.select(dict(type=GlyphRenderer))[0]
    glyph.y_range_name = 'twiny'


def bivariate_plots(data,global_config):
    d1=datetime.now()
    d1=d1.strftime("%m%d%Y_%H%M%S")
    numeric_cols= data.select_dtypes([np.number]).columns.tolist()
    dv=global_config['dv']
    tc=global_config['time_column']
    data[tc]=data[tc].astype('datetime64[ns]')
    numeric_cols=[ x for x in numeric_cols if x not in [tc,dv]]
    plot_dict={'Bivariate Plots':{}}
    for i in numeric_cols:
        plot=data.hvplot.line(x=tc, y=[i,dv], legend='top', height=500, width=950, by=['index.year','index.month'])
        data_f=data.loc[:,[tc,i,dv]]
        data_f=data_f.set_index(tc)
        a=data_f[i].hvplot(yaxis='right',ylim=(0,data_f[i].max()))
        b=data_f[dv].hvplot(yaxis='left', ylim=(0,data_f[dv].max())).opts(hooks=[apply_formatter])
        plot=a*b
        plot=plot.opts(legend_position='top_left',xrotation=90,width=950,height=500,framewise=True, xticks=10, yticks=10,xformatter = DatetimeTickFormatter(months = '%b %Y'))
        plot_dict['Bivariate Plots'][i]={}
        plot_dict['Bivariate Plots'][i]['plot']=plot
    create_report(plot_dict, name=f'eda_bivariate_plots_{global_config["run_name"]}_{d1}', path='Output/eda_output/', format='.html', split_sheets=True, tiger_template=False)
   

#function which calls all above functions
def data_analysis(data,global_config):
    try:
        if global_config['seasonality']['add_col']:
            logging.info('Adding seasonality column')
            data=get_seasonality_column(data,global_config['dv'],global_config['seasonality']['level'],global_config['time_column'])
    except Exception as e:
        logging.error("Exception occurred while adding seasonality column", exc_info=True)
    
    try:
        if global_config['add_trend_column']:
            logging.info('Adding trend column')
            data=add_trend_column(data)
    except Exception as e:
        logging.error("Exception occurred while adding trend column", exc_info=True)
    
    d1=datetime.now()
    d1=d1.strftime("%m%d%Y_%H%M%S")
    
    #getting univariate analysis 
    try:
        x,y=univariate_analysis(data,global_config['time_column'])
        if (x.shape[0]>0) |(y.shape[0]>0):
            logging.info('saving univariate results to a file')
            wb=f'Output/eda_output/univariate_analysis_{global_config["run_name"]}_{d1}.xlsx'
            writer=pd.ExcelWriter(wb)
            x.to_excel(writer,'Numeric_cols',index=False)
            if y.shape[0]>0:
                x.to_excel(writer,'Categorical_cols',index=False)
            writer.close()
    except Exception as e:
        logging.error("Exception occurred while adding running Univariate analysis", exc_info=True)
    
    #saving eda report 
    try:
        eda_report_generation(data,global_config) 
    except Exception as e:
        logging.error("Exception occurred while saving tigerml eda report", exc_info=True)
    #correlation analysis
    try:
        correlation_analysis(data,global_config)
    except Exception as e:
        logging.error("Exception occurred while running correlation analysis", exc_info=True)
    

    #identifying outliers
    try:
        outliers=identify_outliers(data,global_config['time_column'])
        outliers.to_csv(f'Output/eda_output/outlieres_{global_config["run_name"]}_{d1}.csv',index=False)
    except Exception as e:
        logging.error("Exception occurred while identifying outliers", exc_info=True)
    
    
    try:
        bivariate_plots(data,global_config)
    except Exception as e:
        logging.error("Exception occurred while running bivaraite_plots function", exc_info=True)
    
    return data

