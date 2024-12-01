import csv
import pandas as pd
import os
from glob import glob

name = ['AK','AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA','ID','IL','IN','KS','KY','LA','MD','ME','MI','MN','MO','MS','MT','NC','ND','NE','NH','NJ','NM','NV','NY','OH','OR','PA','RI','SC','SD','TN','TX','UT','VA','VT','WA','WI','WV','WY'] 

os.makedirs('states', exist_ok=True)

# Sorting all regions into the states
for i in name:
    state_name = i + '.csv'
    Data = {'party': [],
            'Gender0': [],
            'Gender1':[],
            'session': [],
            'Class0': [],
            'Class1': [],
            'Class2': [],
            'Ethnicity0': [],
            'Ethnicity1': [],
            'Ethnicity2': [],
            'Ethnicity3': [],
            'Age0': [],
            'Age1': [],
            'Age2': [],
            'Age3': [],
            'Earning0': [],
            'Earning1': [],
            'Earning2': [],
            'Military0': [],
            'Military1': [],
            'Education0': [],
            'Education1': [],
            'Education2': [],
            'choice': [],
            'user':[]}
    state = 'states/' + state_name
    Data = pd.DataFrame(Data)
    Data.to_csv(state, mode='w', index=False)
    count = 0
    for s in glob('folder/*.csv'):
        if os.path.basename(s).startswith(i):
            b = pd.read_csv(s)
            
            count2 = count + len(b['user'])
            
            b['user'] = range(count, count2)
            
            count = len(b['user'])
            
            b.to_csv(state, mode='a', index=False, header=False)