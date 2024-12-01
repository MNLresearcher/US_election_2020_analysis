import numpy as np
from time import time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
import subprocess

from sklearn.preprocessing import LabelEncoder

from torch_choice.model import ConditionalLogitModel
from torch_choice.data import ChoiceDataset, JointDataset, utils
from torch_choice import run
from torch_choice.utils.easy_data_wrapper import EasyDatasetWrapper
from matplotlib.backends.backend_pdf import PdfPages

#def print_diet_shape(d):
#    for key val in d.candidates ():
#        if torch.is_tensor(val):
#            print(f'dict.{key}.shape:{val.shape}')

#insert the name of the csv file
votes = pd.read_csv('.csv')

#Technically speaking one could also use the ChoiceDataset model, but the EasyDatasetWrapper is better suited when you apply it on CSV data. 
# All choices were made in the same session

voting_results = EasyDatasetWrapper(
    main_data = votes,
    purchase_record_column='user',
    choice_column='choice',
    item_name_column='party',
    user_index_column='user',
    session_index_column='session',
    user_observable_columns=['Gender0','Gender1','Class0','Class1','Class2','Age0','Age1','Age2','Ethnicity0','Ethnicity1','Ethnicity2','Ethnicity3','Earning0','Earning1','Earning2','Military0','Military1','Education0','Education1','Education2']
    )

dataset = voting_results.choice_dataset

model = ConditionalLogitModel(coef_variation_dict = {'user_Gender0' : 'item-full',
                                                     'user_Gender1' : 'item-full',
                                                     'user_Class0' : 'item-full',
                                                     'user_Class1' : 'item-full',
                                                     'user_Class2' : 'item-full',
                                                     'user_Age0' : 'item-full',
                                                     'user_Age1' : 'item-full',
                                                     'user_Age2' : 'item-full',
                                                     'user_Earning0' : 'item-full',
                                                     'user_Earning1' : 'item-full',
                                                     'user_Earning2' : 'item-full',
                                                     'user_Military0' : 'item-full',
                                                     'user_Military1' : 'item-full',
                                                     'user_Ethnicity0' : 'item-full',
                                                     'user_Ethnicity1' : 'item-full',
                                                     'user_Ethnicity2' : 'item-full',
                                                     'user_Ethnicity3' : 'item-full',
                                                     'user_Education0' : 'item-full',
                                                     'user_Education1' : 'item-full',
                                                     'user_Education2' : 'item-full'
                                                     },
                              num_param_dict = {'user_Gender0' : 1,
                                                'user_Gender1' : 1,
                                                'user_Class0' : 1,
                                                'user_Class1' : 1,
                                                'user_Class2' : 1,
                                                'user_Age0' : 1,
                                                'user_Age1' : 1,
                                                'user_Age2' : 1,
                                                'user_Earning0' : 1,
                                                'user_Earning1' : 1,
                                                'user_Earning2' : 1,
                                                'user_Military0' : 1,
                                                'user_Military1' : 1,
                                                'user_Ethnicity0' : 1,
                                                'user_Ethnicity1' : 1,
                                                'user_Ethnicity2' : 1,
                                                'user_Ethnicity3' : 1,
                                                'user_Education0' : 1,
                                                'user_Education1' : 1,
                                                'user_Education2' : 1
                                                },
                                                num_items=3
                              )


#Since its an iterative process we need to train the model
start_time = time()
command = run(model, dataset, num_epochs=500, learning_rate=0.01, model_optimizer="LBFGS", batch_size=-1)
result = subprocess.command
result.stdout
print('Time taken:', time() - start_time)

# The generated coefficients are only the mean values, there is also a variation of coefficients between the users
Ethnicity0 = model.get_coefficient('user_Ethnicity0[item-full]').numpy()
Ethnicity1 = model.get_coefficient('user_Ethnicity1[item-full]').numpy()
Ethnicity2 = model.get_coefficient('user_Ethnicity2[item-full]').numpy()
Ethnicity3 = model.get_coefficient('user_Ethnicity3[item-full]').numpy()
Education0 = model.get_coefficient('user_Education0[item-full]').numpy()
Education1 = model.get_coefficient('user_Education1[item-full]').numpy()
Education2 = model.get_coefficient('user_Education2[item-full]').numpy()
Class0 = model.get_coefficient('user_Class0[item-full]').numpy()
Class1 = model.get_coefficient('user_Class1[item-full]').numpy()
Class2 = model.get_coefficient('user_Class2[item-full]').numpy()
Military0 = model.get_coefficient('user_Military0[item-full]').numpy()
Military1 = model.get_coefficient('user_Military1[item-full]').numpy()
Gender0 = model.get_coefficient('user_Gender0[item-full]').numpy()
Gender1 = model.get_coefficient('user_Gender1[item-full]').numpy()
Earning0 = model.get_coefficient('user_Earning0[item-full]').numpy()
Earning1 = model.get_coefficient('user_Earning1[item-full]').numpy()
Earning2 = model.get_coefficient('user_Earning2[item-full]').numpy()
Age0 = model.get_coefficient('user_Age0[item-full]').numpy()
Age1 = model.get_coefficient('user_Age1[item-full]').numpy()
Age2 = model.get_coefficient('user_Age2[item-full]').numpy()

# squeezing it turning the coefficients into a one dimensional numpy-array
Ethnicity0 = Ethnicity0.squeeze()
Ethnicity1 = Ethnicity1.squeeze()
Ethnicity2 = Ethnicity2.squeeze()
Ethnicity3 = Ethnicity3.squeeze()
Education0 = Education0.squeeze()
Education1 = Education1.squeeze()
Education2 = Education2.squeeze()
Class0 = Class0.squeeze()
Class1 = Class1.squeeze()
Class2 = Class2.squeeze()
Military0 = Military0.squeeze()
Military1 = Military1.squeeze()
Gender0 = Gender0.squeeze()
Gender1 = Gender1.squeeze()
Earning0 = Earning0.squeeze()
Earning1 = Earning1.squeeze()
Earning2 = Earning2.squeeze()
Age0 = Age0.squeeze()
Age1 = Age1.squeeze()
Age2 = Age2.squeeze()


# plotting it 
plot_names = ['user_Ethnicity0','user_Ethnicity1','user_Ethnicity2','user_Ethnicity3','user_Education0','user_Education1','user_Education2','user_Class0','user_Class1','user_Class2','user_Military0','user_Military1','user_Gender0','user_Gender1','user_Earning0','user_Earning1','user_Earning2','user_Age0','user_Age1','user_Age2']

fix, ax = plt.subplots(figsize=(10, 5))
figs = range(20)
alphabet = [Ethnicity0,Ethnicity1,Ethnicity2,Ethnicity3,Education0,Education1,Education2,Class0,Class1,Class2,Military0,Military1,Gender0,Gender1,Earning0,Earning1,Earning2,Age0,Age1,Age2]
a = 0
os.makedirs('results', exist_ok=True)
for fig in figs:
    ax.hist(alphabet[fig])
    new_name = 'results/'+plot_names[a]+'.png' 
    plt.savefig(new_name)
    a += 1

