import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
import math
import logging 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score
import BayesFramework.model_config_handler as mch
import BayesFramework.Regression as reg
import BayesFramework.plot_utils as plf
import hvplot.pandas
from hvplot import hvPlot
from tigerml.core.reports import create_report
import holoviews as hv
from holoviews import opts
from tigerml.core.scoring import mape,wmape,root_mean_squared_error

#Calculates adjusted r square
def get_adj_r2(rsq,nrows,ncols):
    s=1-rsq
    s1=(nrows-1)/(nrows-ncols-1)
    r=1-(s*s1)
    return r 

#test ttrain split
def trn_test_split(data,model_config):
    cols=model_config['columns']['marketing_vars'].copy()+model_config['columns']['base_vars'].copy()+[model_config['dv'],model_config['time_column']]
    cols=list(set(cols))
    data=data.loc[:,cols] 
    if model_config['test_train']['split_type']=='random':
        X=data.drop(model_config['dv'],axis=1)
        y=data[model_config['dv']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_config['test_train']['test_size'], random_state=42)
        return X_train, X_test, y_train, y_test

    elif model_config['test_train']['split_type']=='sequential':
        nrows=data.shape[0]
        trows=nrows*model_config['test_train']['test_size']
        train=data.iloc[0:trows,:]
        test=data.iloc[trows:,:]
        return train.drop(dv,axis=1),test.drop(dv,axis=1),trian[dv],test[dv]

#scaling 
def scale_variables(X_train,X_test,X,model_config):
    vars_to_scale=model_config['columns']['base_vars']+model_config['columns']['marketing_vars']
    vars_to_scale=list(set(vars_to_scale)-set(model_config['columns']['categorical_vars']))
    rng=model_config['scaling']['col_range']
    rng=rng.split(',')
    rng=[int(x) for x in rng]
    rng=tuple(rng)
    scaler=MinMaxScaler()
    scaler.fit(X_train[vars_to_scale])
    X_train[vars_to_scale]=scaler.transform(X_train[vars_to_scale])
    X_test[vars_to_scale]=scaler.transform(X_test[vars_to_scale])
    X[vars_to_scale]=scaler.transform(X[vars_to_scale])
    return X_train,X_test,X

#log transformation    
def log_transformation(X_train,X_test,X,y_train,y_test,y,model_config):
    vars_to_transform=model_config['columns']['base_vars'].copy()+model_config['columns']['marketing_vars'].copy()
    vars_to_transform=list(set(vars_to_transform)-set(model_config['columns']['categorical_vars']))
    for v in vars_to_transform:
        X_train[v]=np.log(X_train[v]+1)
        X_test[v]=np.log(X_test[v]+1)
        X[v]=np.log(X[v]+1)
    y_train=np.log(y_train+1)
    y_test=np.log(y_test+1)
    y=np.log(y+1)
    return X_train,X_test,X,y_train,y_test,y

#builds lasso model
def build_lasso(X_train,y_train,X_test,y_test,X,y,model_config):
    dv=model_config['dv']
    ts=model_config['time_column']
    X=X.drop([ts],axis=1)
    model=LassoCV()
    model.fit(X_train,y_train)
    coef_df=pd.DataFrame()
    coef_df['column']=X_train.columns
    coef_df['beta']=model.coef_
    i_df=pd.DataFrame()
    i_df['column']=['global_intercept']
    i_df['beta']=[model.intercept_]
    coef_df=coef_df.append(i_df)
    y_train_pred=model.predict(X_train)
    y_test_pred=model.predict(X_test)
    y_pred=model.predict(X)
    return coef_df,y_train_pred,y_test_pred,y_pred

def build_bayesian(X_train,y_train,X_test,y_test,X,y,model_config,global_config):
    train=X_train.copy()
    train[model_config['dv']]=y
    cols=train.columns.tolist()
    cols_dict={x.replace('.','_'):x for x in cols}
    cols_dict['global_intercept']='global_intercept'
    cols_dict['sigma_target']='sigma_target'
    train.columns=[x.replace('.','_') for x in cols]
    test=X_test.copy()
    test[model_config['dv']]=y_test.copy()
    d=X.copy()
    d[model_config['dv']]=y.copy()
    test.columns=[x.replace('.','_') for x in test.columns.tolist()]
    d.columns=[x.replace('.','_') for x in d.columns.tolist()]
    exp_name='new_run'
    run_name=global_config['run_name']
    config_ini_name = ""
    config_excel = model_config['type_of_model']['bayes_config_file']
    get_config_data = mch.Config(config_ini_name)
    model_config_df, framework_config_df = get_config_data.get_config(config_excel)
    model_config_df['IDV']=model_config_df['IDV'].str.replace('.','_')
    model_config_df['DV']=model_config_df['DV'].str.replace('.','_')
    bl = reg.BayesianEstimation(
        train,
        model_config_df,
        framework_config_df,
        experiment_name=exp_name,
        run_name=run_name,
    )
    bl.train()
    bl.summary()
    pl = plf.Plot(bl)
    pl.save_all_plots()
    files=os.listdir(model_config['type_of_model']['bayes_wd'])
    files=[f for f in files if f.split('.')[-1]=='xlsx']
    paths = [os.path.join(model_config['type_of_model']['bayes_wd'], basename) for basename in files]
    req_path=max(paths, key=os.path.getctime)
    print(req_path)
    beta_df=pd.read_excel(req_path,sheet_name='Sheet1')
    pred_results_train, r2score, rms, mapp, ma, wmap = bl.predict(data_pr=train)
    pred_results_test, r2score, rms, mapp, ma, wmap = bl.predict(data_pr=test)
    pred_results, r2score, rms, mapp, ma, wmap = bl.predict(data_pr=d)
    cols=beta_df.columns.tolist()
    cols[0]='column'
    beta_df.columns=cols
    beta_df['column']=beta_df['column'].str.replace('fixed_slope_','')
    beta_df['column']=beta_df['column'].map(cols_dict)
    return beta_df,pred_results_train,pred_results_test,pred_results


#gets model metircs for oth train and test data
def get_metrics(y_train,y_train_pred,y_test,y_test_pred,y,y_pred,X_train,X_test,X,model_config):
    if model_config['type_of_model']['type']=='multiplicative':
        y_train=np.exp(y_train)
        y_train_pred=np.exp(y_train_pred)
        y_test=np.exp(y_test)
        y_test_pred=np.exp(y_test_pred)
        y=np.exp(y)
        y_pred=np.exp(y_pred)
    train_r2=r2_score(y_train,y_train_pred)
    test_r2=r2_score(y_test,y_test_pred)
    rsqu=r2_score(y,y_pred)
    metrics=pd.DataFrame()
    metrics['metric']=['rmse','mape','wmape','adjr2','r2']
    metrics['train']=[root_mean_squared_error(y_train,y_train_pred),(mape(y_train,y_train_pred)*100),(wmape(y_train,y_train_pred)*100),get_adj_r2(train_r2,X_train.shape[0],X_train.shape[1]),train_r2]
    metrics['test']=[root_mean_squared_error(y_test,y_test_pred),(mape(y_test,y_test_pred)*100),(wmape(y_test,y_test_pred)*100),'NA',test_r2]
    metrics['full']=[root_mean_squared_error(y,y_pred),(mape(y,y_pred)*100),(wmape(y,y_pred)*100),get_adj_r2(rsqu,X.shape[0],X.shape[1]),rsqu]
    return metrics

      
#returns exponential values of all variables of a dataframe 
def get_exp_df(df,model_config):
    indicator_cols=model_config['columns']['categorical_vars'].copy()
    cols=[x for x in df.columns.tolist() if (x not in indicator_cols)&(x!=model_config['time_column'])&(x!=model_config['sales_dollars_column'])]
    for c in cols:
        df[c]=df[c].apply(lambda x:math.exp(x))
    return df

#subsets 
def subset_columns(data,model_config):
    cols=model_config['columns']['marketing_vars'].copy()+model_config['columns']['base_vars'].copy()+[model_config['dv'],model_config['time_column'],model_config['sales_dollars_column']]
    cols=list(set(cols))
    data=data.loc[:,cols]
    return data

#reads model config and builds model and gets the results
def get_model_results(data,model_config,global_config):
    try:
        d2=datetime.now()
        d2=d2.strftime("%m%d%Y_%H%M%S")

        data=subset_columns(data,model_config)
        data=data.fillna(0)
        
        X_train, X_test, y_train, y_test=trn_test_split(data,model_config)
        X_train_o, X_test_o, y_train_o, y_test_o=X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
        
        X=data.drop([model_config['dv'],model_config['sales_dollars_column']],axis=1)
        y=data[model_config['dv']]
        X_train.drop(model_config['time_column'],axis=1,inplace=True)
        X_test.drop(model_config['time_column'],axis=1,inplace=True)
        X_train,X_test,X=scale_variables(X_train,X_test,X,model_config)
        
        if model_config['type_of_model']['type']=='multiplicative':
            X_train,X_test,X,y_train,y_test,y=log_transformation(X_train,X_test,X,y_train,y_test,y,model_config)
        
        if model_config['type_of_model']['algo']=='lasso':
            coef_df,y_train_pred,y_test_pred,y_pred=build_lasso(X_train,y_train,X_test,y_test,X,y,model_config)
            pd.to_csv("Lasso_coef_df", coef_df)
        
        if model_config['type_of_model']['algo']=='bayesian':
            coef_df,y_train_pred,y_test_pred,y_pred=build_bayesian(X_train,y_train,X_test,y_test,X,y,model_config,global_config)
            beta_df=coef_df.copy()
            coef_df=coef_df.iloc[:,0:2]
            

        metrics=get_metrics(y_train,y_train_pred,y_test,y_test_pred,y,y_pred,X_train,X_test,X,model_config)
        
        for c in X.columns.tolist():
            data[c]=X[c]
        data[model_config['dv']]=y
        data['y_pred']=y_pred
        data=get_exp_df(data,model_config)
        report_dict={}
        wb=f'Output/model_output/model_output_{global_config["run_name"]}_{d2}_{model_config["type_of_model"]["algo"]}.xlsx'
        writer=pd.ExcelWriter(wb)
        train=X_train_o
        train[model_config['dv']]=y_train_o
        train['y_pred']=np.exp(y_train_pred)
        test=X_test_o
        test[model_config['dv']]=y_test_o
        test['y_pred']=np.exp(y_test_pred) 
        data.to_excel(writer,'full_data',index=False)
        train.to_excel(writer,'train_data',index=False)
        test.to_excel(writer,'test_data',index=False)
        tc=model_config['time_column']
        dv=model_config['dv']
        train_plot=train.hvplot.line(x=tc, y=['y_pred',dv], legend='top', height=500, width=950).opts(legend_position='top_left',xrotation=90)
        test_plot=test.hvplot.line(x=tc, y=['y_pred',dv], legend='top', height=500, width=950).opts(legend_position='top_left',xrotation=90)
        full_plot=data.hvplot.line(x=tc, y=['y_pred',dv], legend='top', height=500, width=950).opts(legend_position='top_left',xrotation=90)
        report_dict['Actual_vs_Predicted']={}
        report_dict['Actual_vs_Predicted']['total_data']={}
        report_dict['Actual_vs_Predicted']['total_data']['plot']=full_plot
        report_dict['Actual_vs_Predicted']['train_data']={}
        report_dict['Actual_vs_Predicted']['train_data']['plot']=train_plot
        report_dict['Actual_vs_Predicted']['test_data']={}
        report_dict['Actual_vs_Predicted']['test_data']['plot']=test_plot    
        if model_config['type_of_model']['algo']=='lasso':
            coef_df.to_excel(writer,'coeficients',index=False)
            coef_df=coef_df.reset_index(drop=True)
            report_dict['coeficients']=coef_df
        elif model_config['type_of_model']['algo']=='bayesian':
            beta_df.to_excel(writer,'coeficients',index=False) 
            beta_df=beta_df.reset_index(drop=True)
            report_dict['coeficients']=beta_df
        metrics.to_excel(writer,'metrics',index=False)
        report_dict['model_metrics']=metrics
        final_report_dict={'Model Output':report_dict}
        
        #final_report_dict={'Model Output':report_dict}
        create_report(report_dict, name=f'model_output_{global_config["run_name"]}_{d2}_{model_config["type_of_model"]["algo"]}', path='Output/model_output/', format='.html', split_sheets=True, tiger_template=False)
        workbook = writer.book
        sheets=['full_data','train_data','test_data']
        for i in sheets:
            if i =='full_data':
                cols=data.columns.tolist()
                last_row=data.shape[0]
                cat_first_col=cols.index(model_config['time_column'])
                val1_first_col=cols.index(model_config['dv'])
                val2_first_col=cols.index('y_pred')
            elif i=='train_data':
                cols=train.columns.tolist()
                last_row=train.shape[0]
                cat_first_col=cols.index(model_config['time_column'])
                val1_first_col=cols.index(model_config['dv'])
                val2_first_col=cols.index('y_pred')
            else:
                cols=test.columns.tolist()
                last_row=test.shape[0]
                cat_first_col=cols.index(model_config['time_column'])
                val1_first_col=cols.index(model_config['dv'])
                val2_first_col=cols.index('y_pred')
            worksheet = writer.sheets[i]
            chart = workbook.add_chart({'type': 'line'})
            chart.set_size({'width': 720, 'height': 300})
            chart.add_series({
                'name':[i,0,val1_first_col],
                'categories' : [i,1,cat_first_col,last_row,cat_first_col],
                'values' :[i,1,val1_first_col,last_row,val1_first_col]})
            chart.add_series({
                'name':[i,0,val2_first_col],
                'categories' : [i,1,cat_first_col,last_row,cat_first_col],
                'values' :[i,1,val2_first_col,last_row,val2_first_col]})
            
            worksheet.insert_chart('E5', chart)
        writer.save()
    except Exception as e:
        logging.error("Exception occurred in modelling part", exc_info=True)
    return data, coef_df, y_pred
    
    



