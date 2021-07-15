import sys
import os 
import pandas as pd
from eda import *
from feature_engineering import *
from modelling import *
from attributions import *
from datetime import datetime,date
import logging 
import yaml
import warnings
warnings.filterwarnings("ignore")
today = date.today()
d1 = today.strftime("%d-%m-%Y")
d2=datetime.now()
d2=d2.strftime("%m%d%Y_%H%M%S")

folders=['Output','logs','Output/eda_output','Output/feature_transformations','Output/feature_transformed_data','Output/model_output','Output/modelled_data','Output/attribution_output']

for f in folders:
    if not os.path.isdir(f):
        os.mkdir(f)

#reading config file
config_file = open("D:/MMX/mmx_ct_python/mmx_ct_python/config/config.yml")
parsed_config_file = yaml.load(config_file, Loader=yaml.FullLoader)
global_config=parsed_config_file['global_config']
feature_config=parsed_config_file['feature_config']
model_config=parsed_config_file['model_config']
attribution_config = parsed_config_file['attribution_config']

#logs configuration
logging.basicConfig(filename=f'logs/log_{global_config["run_name"]}_{d2}.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',level=logging.DEBUG)

#reading data file
data=pd.read_csv(global_config['data_file_location'])


#EDA
if global_config['eda']:
    logging.info('Runing EDA')
    data=data_analysis(data,global_config)



#Feature Transformation
if global_config['feature_transformations']:
    logging.info('Runing Feature Transformation Stage')
    data=feature_eng(data,global_config,feature_config)
    if feature_config['save_file']:
        data.to_csv(f'Output/feature_transformed_data/data_{global_config["run_name"]}_{d1}.csv',index=False)
        

#modelling
if global_config['modelling']:
    data, coef_df, pred = get_model_results(data,model_config,global_config)
    if model_config['save_file']:
        data.to_csv(f'Output/modelled_data/model_data_{global_config["run_name"]}_{d1}.csv',index=False)
        coef_df.to_csv(f'Output/modelled_data/coefficients_{global_config["run_name"]}_{d1}.csv',index=False)
        pd.DataFrame(pred).to_csv(f'Output/modelled_data/predictions_{global_config["run_name"]}_{d1}.csv',index=False)


    
    
#Attribution calculation
if global_config['attribution']:
#if model_config['contributions']['Indicator']:
    logging.info('Runing Attribution calculations')
    get_attributions(data, coef_df,pred, attribution_config, global_config)
    
    
logging.info('End of Analysis')


