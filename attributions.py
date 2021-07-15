import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
import math
import logging 
import yaml
from datetime import datetime,date
from tigerml.core.reports import create_report
from sklearn.preprocessing import MinMaxScaler



#raw contribution calculation
def contrs(y_mar,x_mar):
        return y_mar-(y_mar/x_mar)
    
    
#scaled contribution calculation   
def scaled_contr(y_mar,summ,x_mar):
    return x_mar*(y_mar/summ)


#Contributions at row level
def columns_contributions_row_level(pred_df,beta_df,indicator_cols,attribution_config,pred_col_name):
    beta_df['column']=beta_df['column'].str.replace('fixed_slope_','')
    idvs=beta_df['column'].tolist()
    pred_cols=pred_df.columns.tolist()
    idvs=list(set(idvs)&set(pred_cols))
    pred_df_idvs=pred_df.loc[:,idvs]
    indicator_cols=attribution_config['columns']['categorical_vars'].copy()
    marketing_cols=attribution_config['columns']['marketing_vars'].copy()
    control_cols=attribution_config['columns']['base_vars'].copy()
    control_i_cols=[x for x in pred_df_idvs.columns.tolist() if (x in marketing_cols) & (x in indicator_cols)]
    marketing_i_cols=[x for x in pred_df_idvs.columns.tolist() if (x in control_cols)& (x in indicator_cols)]
    beta_dict={}
    for i in range(beta_df.shape[0]):
        row=beta_df.iloc[i,:]
        beta_dict[row['column']]=row['mean']
    for i in pred_df_idvs.columns.tolist():
        if i not in indicator_cols:
            pred_df_idvs[i]=pred_df_idvs[i].apply(lambda x:x**beta_dict[i])
        elif i in indicator_cols:
            pred_df_idvs[i]=pred_df_idvs[i].apply(lambda x:x*beta_dict[i])
            pred_df_idvs[i]=pred_df_idvs[i].apply(lambda x:math.exp(x))
    pred_df_idvs['y_pred']=pred_df[pred_col_name]
    pred_df_idvs['y_con']=1
    for c in control_cols:
        pred_df_idvs['y_con']=pred_df_idvs['y_con']*pred_df_idvs[c]
    for ci in control_i_cols:
        pred_df_idvs['y_con']=pred_df_idvs['y_con']*pred_df_idvs[ci]
    pred_df_idvs['y_con']=pred_df_idvs['y_con']*math.exp(beta_dict['global_intercept'])
    pred_df_idvs['y_mar']=pred_df_idvs['y_pred']-pred_df_idvs['y_con']
    return pred_df_idvs,beta_dict



#returns base sales and marketing sales given input data, coefs, prediction 
def base_marketing_split(pred_df,beta_df,attribution_config):
    indicator_cols=attribution_config['columns']['categorical_vars'].copy()
    pred_col_name='y_pred'
    pred_df_idvs,beta_dict=columns_contributions_row_level(pred_df,beta_df,indicator_cols,attribution_config,pred_col_name)
    return pred_df_idvs['y_con'].sum(),pred_df_idvs['y_mar'].sum()
    

#returns individual contributions of all marketing variables it takes input data, prediction, coefs as input  
def individual_contributions(pred_df,beta_df,attribution_config,pred_col_name):
    indicator_cols=attribution_config['columns']['categorical_vars'].copy()
    marketing_cols=attribution_config['columns']['marketing_vars'].copy()
    control_cols=attribution_config['columns']['base_vars'].copy()
    control_i_cols=[x for x in control_cols if (x in control_cols) & (x in indicator_cols)]
    marketing_i_cols=[x for x in marketing_cols if (x in marketing_cols)& (x in indicator_cols)]
    pred_col_name='y_pred'
    pred_df_idvs,beta_dict=columns_contributions_row_level(pred_df,beta_df,indicator_cols,attribution_config,pred_col_name)
    pred_df_idvs_m=pred_df_idvs.loc[:,marketing_cols]
    pred_df_idvs_m['y_mar']=pred_df_idvs['y_mar']
    for m in marketing_cols:
        pred_df_idvs_m[f'{m}_raw_contr']=pred_df_idvs_m.apply(lambda row:contrs(row['y_mar'],row[m]),axis=1)
    pred_df_idvs_m=pred_df_idvs_m.loc[:,[x for x in pred_df_idvs_m.columns.tolist() if x.split('_')[-1]=='contr']]
    raw_sum_df_m=pd.DataFrame()
    raw_sum_df_m['column']=pred_df_idvs_m.columns
    raw_sum_df_m['raw_contr']=raw_sum_df_m['column'].apply(lambda x:pred_df_idvs_m[x].sum())
    pred_df_idvs_m['sum']=pred_df_idvs_m.sum(axis=1)
    pred_df_idvs_m['y_mar']=pred_df_idvs['y_mar']
    
    for m in marketing_cols:
        pred_df_idvs_m[f'{m}_scale_contr']=pred_df_idvs_m.apply(lambda row:scaled_contr(row['y_mar'],row['sum'],row[f'{m}_raw_contr']),axis=1)
    pred_df_idvs_m=pred_df_idvs_m.loc[:,[x for x in pred_df_idvs_m.columns.tolist() if 'scale_contr' in x]]
    sum_df_m=pd.DataFrame()
    sum_df_m['column']=pred_df_idvs_m.columns.tolist()
    sum_df_m['scaled_contr']=sum_df_m['column'].apply(lambda x:pred_df_idvs_m[x].sum())
    sum_df_m['column']=sum_df_m['column'].str.replace('_scale_contr','')
    raw_sum_df_m['column']=raw_sum_df_m['column'].str.replace('_raw_contr','')
    sum_df_m=sum_df_m.merge(raw_sum_df_m,on='column')
    sum_df_m['per_contr']=(sum_df_m['scaled_contr']/(sum_df_m['scaled_contr'].sum()))
    sum_df_m['per_contr']=sum_df_m['per_contr']*100
    control_cols.extend(control_i_cols)
    pred_df_idvs_c=pred_df_idvs.loc[:,control_cols]
    pred_df_idvs_c['intercept']=math.exp(beta_dict['global_intercept'])
    pred_df_idvs_c['y_con']=pred_df_idvs['y_con']
    control_cols.append('intercept')
    for c in control_cols:
        pred_df_idvs_c[f'{c}_raw_contr']=pred_df_idvs_c.apply(lambda row:contrs(row['y_con'],row[c]),axis=1)
    pred_df_idvs_c=pred_df_idvs_c.loc[:,[x for x in pred_df_idvs_c.columns.tolist() if x.split('_')[-1]=='contr']]
    raw_sum_df_c=pd.DataFrame()
    raw_sum_df_c['column']=pred_df_idvs_c.columns
    raw_sum_df_c['raw_contr']=raw_sum_df_c['column'].apply(lambda x:pred_df_idvs_c[x].sum())
    pred_df_idvs_c['sum']=pred_df_idvs_c.sum(axis=1)
    pred_df_idvs_c['y_con']=pred_df_idvs['y_con']
    for c in control_cols:
        pred_df_idvs_c[f'{c}_scale_contr']=pred_df_idvs_c.apply(lambda row:scaled_contr(row['y_con'],row['sum'],row[f'{c}_raw_contr']),axis=1)
    pred_df_idvs_c=pred_df_idvs_c.loc[:,[x for x in pred_df_idvs_c.columns.tolist() if 'scale_contr' in x]]
    sum_df_c=pd.DataFrame()
    sum_df_c['column']=pred_df_idvs_c.columns.tolist()
    sum_df_c['scaled_contr']=sum_df_c['column'].apply(lambda x:pred_df_idvs_c[x].sum())
    sum_df_c['column']=sum_df_c['column'].str.replace('_scale_contr','')
    raw_sum_df_c['column']=raw_sum_df_c['column'].str.replace('_raw_contr','')
    sum_df_c=sum_df_c.merge(raw_sum_df_c,on='column')
    sum_df_c['per_contr']=(sum_df_c['scaled_contr']/(sum_df_c['scaled_contr'].sum()))
    sum_df_c['per_contr']=sum_df_c['per_contr']*100
    pred_df_idvs_m['pos_qty']=pred_df[attribution_config['dv']]
    pred_df_idvs_m['y_pred']=pred_df['y_pred'] 
    pred_df_idvs_m['sales']=pred_df[attribution_config['sales_dollars_column']]
    pred_df_idvs_m[attribution_config['time_column']]=pred_df[attribution_config['time_column']]
    return sum_df_c,sum_df_m,pred_df_idvs_m


# returns sales quantity contribution and sales dollars contribution at various levels 
def sales_qty_contr_sums(pred_df,beta_df,pred_contr_df,attribution_config,level):
    pred_contr_df[attribution_config['time_column']]=pd.to_datetime(pred_contr_df[attribution_config['time_column']],format='%Y-%m-%d')
    pred_contr_df['year']=pred_contr_df['date_sunday'].dt.year
    pred_contr_df['month']=pred_contr_df['date_sunday'].dt.month
    pred_contr_df['c_avg_price']=pred_contr_df['sales']/pred_contr_df['pos_qty']
    pred_contr_df.columns=[x.replace('scale_contr','qty_contr')for x in pred_contr_df.columns.tolist()]
    qty_cols=[x for x in pred_contr_df.columns.tolist() if '_qty_contr' in x]
    for c in qty_cols:
        sc=c.replace('qty_contr','sales_contr')
        pred_contr_df[sc]=pred_contr_df[c]*pred_contr_df['c_avg_price']
    cols=pred_contr_df.columns.tolist()
    cols=[c for c in cols if ('qty_contr' in c) |('sales_contr' in c)]
    overall_sums=pd.DataFrame()
    overall_sums['column']=cols
    overall_sums['overall_contribution']=overall_sums['column'].apply(lambda x:pred_contr_df[x].sum())
    overall_sums=overall_sums.loc[overall_sums['column'].str.endswith('sales_contr')]
    return overall_sums


#returns aggregated spends of all marketing channels at various levels
def spend_sums(attribution_config,level):
    spend_data=pd.read_csv(attribution_config['contributions']['spends_file_location'])
    spend_data[attribution_config['time_column']]=pd.to_datetime(spend_data[attribution_config['time_column']],infer_datetime_format=True)
    spend_data['year']=spend_data['date_sunday'].dt.year
    spend_data['month']=spend_data['date_sunday'].dt.month
    spend_data['quarter']=spend_data['date_sunday'].dt.quarter
    spend_data['quarter']='Q'+spend_data['quarter'].astype(str)
    if attribution_config['contributions']['Level']=='Monthly':
        spend_data['level']=spend_data['year'].astype(str)+'-'+spend_data['month'].astype(str)
    elif attribution_config['contributions']['Level']=='Quarterly':
        spend_data['level']=spend_data['year'].astype(str)+'-'+spend_data['quarter'].astype(str)
    elif attribution_config['contributions']['Level']=='Yearly':
        spend_data['level']=spend_data['year'].astype(str)
    elif attribution_config['contributions']['Level']=='Overall':
        spend_data['level']='Overall'
    
    spend_data=spend_data.loc[spend_data['level']==level,:]
    cols=spend_data.columns.tolist()
    cols=[c for c in cols if c!=attribution_config['time_column']]
    overall_sums=pd.DataFrame()
    overall_sums['column']=cols
    overall_sums['overall_spend']=overall_sums['column'].apply(lambda x:spend_data[x].sum())
    return overall_sums


#intermediate functon
def get_spend(spend_data,x,column_name):
    x=x.replace('_sales_contr','')
    if x in spend_data['column'].tolist():
        s=spend_data.loc[spend_data['column']==x,:][column_name]
        if s.shape[0]>0:
            return s.tolist()[0]
    else:
        prefix=x.split('.')[0]
        spend_data['prefix']=spend_data['column'].apply(lambda x: x.split('.')[0])
        s=spend_data.loc[spend_data['prefix']==prefix,:][column_name]
        if s.shape[0]>0:
            return s.tolist()[0]
        else:
            return None

def get_response_curves_data(quantity_contributions_my,multipliers):
    quantity_contributions_my['column']=quantity_contributions_my['column'].str.replace('_csales_contr','')
    cols=quantity_contributions_my['column'].unique().tolist()
    final_response_df=pd.DataFrame()
    for c in cols:
        response_df=pd.DataFrame()
        response_df['multiplier']=multipliers
        response_df['touch_point']=c
        response_df['beta']=quantity_contributions_my.loc[quantity_contributions_my['column']==c,'mean'].tolist()[0]
        response_df['sales']=quantity_contributions_my.loc[quantity_contributions_my['column']==c,'overall_contribution'].tolist()[0]
        response_df['spend']=quantity_contributions_my.loc[quantity_contributions_my['column']==c,'overall_spend'].tolist()[0]
        response_df['new_spend']=response_df['spend']*(1+response_df['multiplier'])
        response_df['spend_change']=response_df['new_spend']-response_df['spend']
        response_df['new_sales']=(((1+response_df['new_spend'])/response_df['spend'])**response_df['beta'])*response_df['sales']
        final_response_df=final_response_df.append(response_df)
    return final_response_df

def get_contributions_at_level(attribution_config,data,coef_df):
    if attribution_config['contributions']['Level']=='Monthly':
        data[attribution_config['time_column']]=pd.to_datetime(data[attribution_config['time_column']],infer_datetime_format=True)
        data['month_num']=data[attribution_config['time_column']].dt.month
        data['year']=data[attribution_config['time_column']].dt.year
        data['level']=data['year'].astype(str)+'-'+data['month_num'].astype(str)
    elif attribution_config['contributions']['Level']=='Quarterly':
        data[attribution_config['time_column']]=pd.to_datetime(data[attribution_config['time_column']],infer_datetime_format=True)
        data['quarter']=data[attribution_config['time_column']].dt.quarter
        data['quarter']='Q'+data['quarter'].astype(str)
        data['year']=data[attribution_config['time_column']].dt.year
        data['level']=data['year'].astype(str)+'-'+data['quarter'].astype(str)
    elif attribution_config['contributions']['Level']=='Yearly':
        data[attribution_config['time_column']]=pd.to_datetime(data[attribution_config['time_column']],infer_datetime_format=True)
        data['year']=data[attribution_config['time_column']].dt.year
        data['level']=data['year'].astype(str)
    elif attribution_config['contributions']['Level']=='Overall':
        data[attribution_config['time_column']]=pd.to_datetime(data[attribution_config['time_column']],infer_datetime_format=True)
        data['year']=data[attribution_config['time_column']].dt.year
        data['level']='Overall'
    uniq_levels=data['level'].unique().tolist()
    split_df=pd.DataFrame()
    split_df['split']=['Base Split','Marketing Split']
    sum_df_c_final=pd.DataFrame()
    sum_df_m_final=pd.DataFrame()
    qty_contributions_final=pd.DataFrame()
    for m in uniq_levels:
        data_f=data.loc[data['level']==m,:]
        co,ma=base_marketing_split(data_f,coef_df,attribution_config)
        split_df[m]=[co,ma]
        split_df[f'per_{m}']=(split_df[m]/split_df[m].sum())*100
        sum_df_c,sum_df_m,pred_df_idvs_m=individual_contributions(data_f,coef_df,attribution_config,'y_pred')
        sum_df_c.drop(['raw_contr'],axis=1,inplace=True)
        sum_df_m.drop(['raw_contr'],axis=1,inplace=True)
        sum_df_c.rename(columns={'scaled_contr':f'{m}_contr','per_contr':f'{m}_per'},inplace=True)
        sum_df_m.rename(columns={'scaled_contr':f'{m}_contr','per_contr':f'{m}_per'},inplace=True)
        qty_contributions=sales_qty_contr_sums(data_f,coef_df,pred_df_idvs_m,attribution_config,m)
        spend_data=spend_sums(attribution_config,m)
        qty_contributions['overall_spend']=qty_contributions['column'].apply(lambda x:get_spend(spend_data,x,'overall_spend'))
        qty_contributions[f'ROI_{m}']=qty_contributions['overall_contribution']/qty_contributions['overall_spend']
        if attribution_config['contributions']['Level']!='Overall':
            qty_contributions=qty_contributions.loc[:,['column',f'ROI_{m}']]
        if sum_df_c_final.shape[0]==0:
            sum_df_c_final=sum_df_c
        else:
            sum_df_c_final=sum_df_c_final.merge(sum_df_c,on='column',how='outer')
        if sum_df_m_final.shape[0]==0:
            sum_df_m_final=sum_df_m
        else:
            sum_df_m_final=sum_df_m_final.merge(sum_df_m,on='column',how='outer')     
        if qty_contributions_final.shape[0]==0:
            qty_contributions_final=qty_contributions
        else:
            qty_contributions_final=qty_contributions_final.merge(qty_contributions,on='column',how='outer')    
    return split_df,sum_df_c_final,sum_df_m_final,qty_contributions_final,pred_df_idvs_m


def get_attributions(data, coef_df,y_pred, attribution_config, global_config):
    d2=datetime.now()
    d2=d2.strftime("%m%d%Y_%H%M%S")
    try:
        d2=datetime.now()
        d2=d2.strftime("%m%d%Y_%H%M%S")
        
        report_dict={}
        wb=f'Output/attribution_output/Attributions_output_{global_config["run_name"]}_{d2}.xlsx'
        writer=pd.ExcelWriter(wb)
        #if model_config['contributions']['Indicator']:
        split_df,sum_df_c_final,sum_df_m_final,qty_contributions_final,pred_df_idvs_m=get_contributions_at_level(attribution_config,data,coef_df)
        
        attribution_config_overall=attribution_config.copy()
        attribution_config_overall['contributions']['Level']='Overall'
        split_df_o,sum_df_c_final_o,sum_df_m_final_o,qty_contributions_final_o,pred_df_idvs_m=get_contributions_at_level(attribution_config_overall,data,coef_df)
        qty_contributions_final_o.rename(columns={'overall_spend':'spend','ROI_Overall':'ROI','overall_contribution':'sales'},inplace=True)
        
        qty_contributions_final_o['spend_per']=qty_contributions_final_o['spend']/(qty_contributions_final_o['spend'].sum())
        qty_contributions_final_o['sales_per']=qty_contributions_final_o['sales']/(qty_contributions_final_o['sales'].sum())
        qty_contributions_final_o['Efficiency']=qty_contributions_final_o['sales_per']/qty_contributions_final_o['spend_per']
        
        split_df=split_df.set_index('split')
        split_df=split_df.T
        split_df=split_df.reset_index()
        split_df.rename(columns={'index':'split'},inplace=True)
        split_df_actuals=split_df.loc[~split_df['split'].str.startswith('per_'),:]
        split_df_actuals=split_df_actuals.reset_index(drop=True)
        split_df_actuals.rename(columns={'split':'time_period'},inplace=True)
        split_df_per=split_df.loc[split_df['split'].str.startswith('per_'),:]
        split_df_per['split']=split_df_per['split'].str.replace('per_','')
        split_df_per1=split_df_per.copy()
        split_df_per1=split_df_per1.reset_index(drop=True)
        split_df_per1.rename(columns={'split':'time_period'},inplace=True)
        split_df_per1['time_period']=split_df_per1['time_period'].str.replace('per_','')
        split_df_per=split_df_per.melt(id_vars=['split'],var_name='split_type',value_name='percentage')
        split_df_per.rename(columns={'split':'time_period','split_type':'split'},inplace=True)
        split_df_per=split_df_per.sort_values(by=['time_period','split'],ascending=True)
        split_df_per=split_df_per.groupby(['time_period','split']).agg({'percentage':'mean'})
        
        split_df_actuals.to_excel(writer,'split',index=False)
        split_df_per1.to_excel(writer,'split',index=False,startcol=0, startrow=split_df_actuals.shape[0]+3)
        report_dict['Base_vs_Marekting_split']={}
        report_dict['Base_vs_Marekting_split']['Actuals']={}
        report_dict['Base_vs_Marekting_split']['percentage']={}
        report_dict['Base_vs_Marekting_split']['Actuals']['data']=split_df_actuals
        report_dict['Base_vs_Marekting_split']['Actuals']['plot']=split_df_actuals.hvplot.area(x='time_period', y=['Base Split','Marketing Split'], label='Base_vs_Marketing_Split_Actuals',
            width=500, height=500).opts(legend_position='top_left')
        report_dict['Base_vs_Marekting_split']['percentage']['data']=split_df_per1
        report_dict['Base_vs_Marekting_split']['percentage']['plot']=split_df_per.hvplot.bar(stacked=True, label='Base_vs_Marketing_Percenatge',width=500, height=500).opts(legend_position='top_left')
        sum_df_c_final=sum_df_c_final.set_index('column')
        sum_df_c_final=sum_df_c_final.T
        sum_df_c_final=sum_df_c_final.reset_index()
        sum_df_c_final.rename(columns={'index':'column'},inplace=True)
        sum_df_c_final_contr=sum_df_c_final.loc[sum_df_c_final['column'].str.contains('_contr'),:]
        sum_df_c_final_contr['column']=sum_df_c_final_contr['column'].str.replace('_contr','')
        sum_df_c_final_per=sum_df_c_final.loc[sum_df_c_final['column'].str.contains('_per'),:]
        sum_df_c_final_per['column']=sum_df_c_final_per['column'].str.replace('_per','')
        sum_df_c_final_contr=sum_df_c_final_contr.reset_index(drop=True)
        sum_df_c_final_contr.rename(columns={'column':'time_period'},inplace=True)
        sum_df_c_final_per=sum_df_c_final_per.reset_index(drop=True)
        sum_df_c_final_per.rename(columns={'column':'time_period'},inplace=True)
        sum_df_c_final_per1=sum_df_c_final_per.melt(id_vars=['time_period'],var_name='variable',value_name='value')
        sum_df_c_final_per1=sum_df_c_final_per1.sort_values(by=['time_period','variable'],ascending=True)
        sum_df_c_final_per1=sum_df_c_final_per1.groupby(['time_period','variable']).agg({'value':'mean'})
        sum_df_c_final_contr.to_excel(writer,'control_vars_contribution',index=False,startcol=0,startrow=0)
        sum_df_c_final_per.to_excel(writer,'control_vars_contribution',index=False,startcol=0,startrow=sum_df_c_final_contr.shape[0]+3)
        
        report_dict['control_variables_contribution_distribution']={}
        report_dict['control_variables_contribution_distribution']['actuals']={}
        report_dict['control_variables_contribution_distribution']['percentage']={}
        report_dict['control_variables_contribution_distribution']['actuals']['data']=sum_df_c_final_contr
        report_dict['control_variables_contribution_distribution']['actuals']['plot']=sum_df_c_final_contr.hvplot.line(x='time_period', y=[ x for x in sum_df_c_final_contr.columns.tolist() if x!='time_period'], label='control_variables_contribution_distribution_actuals',
            width=500, height=500).opts(legend_position='top_left')
        report_dict['control_variables_contribution_distribution']['percentage']['data']=sum_df_c_final_per
        report_dict['control_variables_contribution_distribution']['percentage']['plot']=sum_df_c_final_per1.hvplot.bar(stacked=True, label='control_variables_contribution_distribution_percentage',width=500, height=500).opts(legend_position='top_left')
        
        sum_df_m_final=sum_df_m_final.set_index('column')
        sum_df_m_final=sum_df_m_final.T
        sum_df_m_final=sum_df_m_final.reset_index()
        sum_df_m_final.rename(columns={'index':'column'},inplace=True)
        sum_df_m_final_contr=sum_df_m_final.loc[sum_df_m_final['column'].str.contains('_contr')]
        sum_df_m_final_contr['column']=sum_df_m_final_contr['column'].str.replace('_contr','')
        sum_df_m_final_per=sum_df_m_final.loc[sum_df_m_final['column'].str.contains('_per'),:]
        sum_df_m_final_per['column']=sum_df_m_final_per['column'].str.replace('_per','')
        sum_df_m_final_contr=sum_df_m_final_contr.reset_index(drop=True)
        sum_df_m_final_contr.rename(columns={'column':'time_period'},inplace=True)
        sum_df_m_final_per=sum_df_m_final_per.reset_index(drop=True)
        sum_df_m_final_per.rename(columns={'column':'time_period'},inplace=True)
        sum_df_m_final_per1=sum_df_m_final_per.melt(id_vars=['time_period'],var_name='variable',value_name='value')
        sum_df_m_final_per1=sum_df_m_final_per1.sort_values(by=['time_period','variable'],ascending=True)
        sum_df_m_final_per1=sum_df_m_final_per1.groupby(['time_period','variable']).agg({'value':'mean'})
        sum_df_m_final_contr.to_excel(writer,'marketing_vars_contribution',index=False,startcol=0,startrow=0)
        sum_df_m_final_per.to_excel(writer,'marketing_vars_contribution',index=False,startcol=0,startrow=sum_df_m_final_contr.shape[0]+3)

        
        report_dict['Marketing_variables_contribution_distribution']={}
        report_dict['Marketing_variables_contribution_distribution']['actuals']={}
        report_dict['Marketing_variables_contribution_distribution']['percentage']={}
        report_dict['Marketing_variables_contribution_distribution']['actuals']['data']=sum_df_m_final_contr
        report_dict['Marketing_variables_contribution_distribution']['actuals']['plot']=sum_df_m_final_contr.hvplot.line(x='time_period', y=[x for x in  sum_df_m_final_contr.columns.tolist() if x!='time_period'], label='Marketing_variables_contribution_distribution_actuals',
            width=500, height=500).opts(legend_position='top_left')
        report_dict['Marketing_variables_contribution_distribution']['percentage']['data']=sum_df_m_final_per
        report_dict['Marketing_variables_contribution_distribution']['percentage']['plot']=sum_df_m_final_per1.hvplot.bar(stacked=True, label='Marketing_variables_contribution_distribution_percentage',width=500, height=500).opts(legend_position='top_left')
        qty_contributions_final=qty_contributions_final.set_index('column')
        qty_contributions_final=qty_contributions_final.T
        qty_contributions_final=qty_contributions_final.reset_index()
        qty_contributions_final.rename(columns={'index':'column'},inplace=True)
        qty_contributions_final['column']=qty_contributions_final['column'].str.replace('ROI_','')
        qty_contributions_final=qty_contributions_final.reset_index(drop=True)
        qty_contributions_final.rename(columns={'column':'time_period'},inplace=True)
        report_dict['Return On Investment']={}
        report_dict['Return On Investment']['data']=qty_contributions_final
        report_dict['Return On Investment']['plot']=qty_contributions_final.hvplot.line(x='time_period', y=[x for x in qty_contributions_final.columns.tolist() if x!='time_period'], label='ROI',width=1000, height=500).opts(legend_position='top_left')
        qty_contributions_final.to_excel(writer,'ROI',index=False)
        pred_df_idvs_m.to_excel(writer,'Raw Marketing Contributions',index=False)

        data['year']=data['year'].astype(int)
        max_year=data['year'].max()
        data_max_year=data.loc[data['year']==max_year,:]
        attribution_config_copy=attribution_config.copy()
        attribution_config_copy['contributions']['Level']='Yearly'

        sum_df_cmy,sum_df_mmy,pred_df_idvs_mmy=individual_contributions(data_max_year,coef_df,attribution_config_copy,'y_pred')
        
        qty_contributions_my=sales_qty_contr_sums(data_max_year,coef_df,pred_df_idvs_mmy,attribution_config_copy,str(max_year))
        spend_data_my=spend_sums(attribution_config_copy,'2020')
        qty_contributions_my['overall_spend']=qty_contributions_my['column'].apply(lambda x:get_spend(spend_data_my,x,'overall_spend'))
        qty_contributions_my['column']=qty_contributions_my['column'].str.replace('_sales_contr','')
        qty_contributions_my=qty_contributions_my.merge(coef_df,on='column',how='left')
        
        multipliers=[round(-0.9+(x*0.1),2) for x in range(0,30)]
        rcd=get_response_curves_data(qty_contributions_my,multipliers)
        
        rcd.to_excel(writer,'Response curves calculation',index=False)
        report_dict['Response Curves']={}
        cols=rcd['touch_point'].unique().tolist()
        for c in cols:
            rcd_f=rcd.loc[rcd['touch_point']==c,:]
            report_dict['Response Curves'][c]={}
            report_dict['Response Curves'][c]['plot']=rcd_f.hvplot.line(x='spend_change', y=['new_sales'], label=f'Response Curve for {c}',width=700, height=500)
        
        final_report_dict={'Model Output':report_dict}
        create_report(report_dict, name=f'Attributions_output_{global_config["run_name"]}_{d2}', path='Output/attribution_output/', format='.html', split_sheets=True, tiger_template=False)
        workbook = writer.book
        
        writer.save()
           
    except Exception as e:
        logging.error("Exception occurred in Attribution calculations", exc_info=True)
