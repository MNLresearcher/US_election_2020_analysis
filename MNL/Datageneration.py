import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from scipy import stats
import h5py
import os
import pickle
from glob import glob
a = pd.read_csv('2020-by-region.csv')

os.makedirs('folder', exist_ok=True)

File = []
Filenames = []
for x in os.listdir('SORT_DIR'):
    if x.endswith(".pq"):
        File.append(x.replace('feats_', ''))
for i in File:
    Filenames.append(i.replace('.pq', ''))

i=0
count = 0
for s in glob('SORT_DIR/*.pq'):
    b = pd.read_parquet(s)
    df = pd.DataFrame([])
    dfs=pd.DataFrame([])
    party = []
    region = []
    Gender = []
    Ethnicity = []
    Age = []
    Earning = []
    Military = []
    Class = []
    Education = []
    choice = []
    for index, row in a.iterrows():
        if row[0]==Filenames[i]:
            party.extend(['votes_D']*row[1])
            party.extend(['votes_R']*row[2])
            party.extend(['votes_oth']*row[3])
            region.extend([0]*sum([row[1],row[2],row[3]]))
            meanAge = b[['AGEP']].mean()
            
            stdAge = b[['AGEP']].std()
            
            Age.extend(np.round(np.random.normal(loc=meanAge, scale=stdAge, size=sum([row[1],row[2],row[3]]))))
            
            meanEarning = b['PERNP'].mean()
            
            stdEarning = b['PERNP'].std()
            
            Earning.extend(np.round(np.random.normal(loc=meanEarning, scale=stdEarning, size=sum([row[1],row[2],row[3]]))))
            
            pSex= [b['SEX'].value_counts()['1']/b['SEX'].count(),b['SEX'].value_counts()['2']/b['SEX'].count()]
            
            p_Class1 = b['COW'].value_counts()['1']/b['COW'].count()
            
            p_Class2 = b['COW'].value_counts()['2']/b['COW'].count()
            
            p_Class3 = b['COW'].value_counts()['3']/b['COW'].count()
            
            p_Class4 = b['COW'].value_counts()['4']/b['COW'].count()
            
            p_Class5 = b['COW'].value_counts()['5']/b['COW'].count()
            
            p_Class6 = b['COW'].value_counts()['6']/b['COW'].count()
            
            p_Class7 = b['COW'].value_counts()['7']/b['COW'].count()
            
            p_Class8 = b['COW'].value_counts()['8']/b['COW'].count()
            
            p_Class9 = b['COW'].value_counts()['9']/b['COW'].count()
            
            pClass = [p_Class1,p_Class2,p_Class3,p_Class4,p_Class5,p_Class6,p_Class7,p_Class8,p_Class9]
            
            p_MIL1 = b['MIL'].value_counts()['1']/b['MIL'].count()
            
            p_MIL2 = b['MIL'].value_counts()['2']/b['MIL'].count()
            
            p_MIL3 = b['MIL'].value_counts()['3']/b['MIL'].count()
            
            p_MIL4 = b['MIL'].value_counts()['4']/b['MIL'].count()
            
            pMilitary = [p_MIL1, p_MIL2, p_MIL3, p_MIL4]
            
            p_SCHL1 = b['SCHL'].value_counts()['01']/b['SCHL'].count()
            
            p_SCHL2 = b['SCHL'].value_counts()['02']/b['SCHL'].count()
            
            p_SCHL3 = b['SCHL'].value_counts()['03']/b['SCHL'].count()
            
            p_SCHL4 = b['SCHL'].value_counts()['04']/b['SCHL'].count()
            
            p_SCHL5 = b['SCHL'].value_counts()['05']/b['SCHL'].count()
            
            p_SCHL6 = b['SCHL'].value_counts()['06']/b['SCHL'].count()
            
            p_SCHL7 = b['SCHL'].value_counts()['07']/b['SCHL'].count()
            
            p_SCHL8 = b['SCHL'].value_counts()['08']/b['SCHL'].count()
            
            p_SCHL9 = b['SCHL'].value_counts()['09']/b['SCHL'].count()
            
            p_SCHL10 = b['SCHL'].value_counts()['10']/b['SCHL'].count()
            
            p_SCHL11 = b['SCHL'].value_counts()['11']/b['SCHL'].count()
            
            p_SCHL12 = b['SCHL'].value_counts()['12']/b['SCHL'].count()
            
            p_SCHL13 = b['SCHL'].value_counts()['13']/b['SCHL'].count()
            
            p_SCHL14 = b['SCHL'].value_counts()['14']/b['SCHL'].count()
            
            p_SCHL15 = b['SCHL'].value_counts()['15']/b['SCHL'].count()
            
            p_SCHL16 = b['SCHL'].value_counts()['16']/b['SCHL'].count()
            
            p_SCHL17 = b['SCHL'].value_counts()['17']/b['SCHL'].count()
            
            p_SCHL18 = b['SCHL'].value_counts()['18']/b['SCHL'].count()
            
            p_SCHL19 = b['SCHL'].value_counts()['19']/b['SCHL'].count()
            
            p_SCHL20 = b['SCHL'].value_counts()['20']/b['SCHL'].count()
            
            p_SCHL21 = b['SCHL'].value_counts()['21']/b['SCHL'].count()
            
            p_SCHL22 = b['SCHL'].value_counts()['22']/b['SCHL'].count()
            
            p_SCHL23 = b['SCHL'].value_counts()['23']/b['SCHL'].count()
            
            p_SCHL24= b['SCHL'].value_counts()['24']/b['SCHL'].count()
            
            pEducation = [p_SCHL1,p_SCHL2,p_SCHL3,p_SCHL4,p_SCHL5,p_SCHL6,p_SCHL7,p_SCHL8,p_SCHL9,p_SCHL10,p_SCHL11,p_SCHL12,p_SCHL13,p_SCHL14,p_SCHL15,p_SCHL16,p_SCHL17,p_SCHL18,p_SCHL19,p_SCHL20,p_SCHL21,p_SCHL22,p_SCHL23,p_SCHL24]
            
            p_Race1 = b['RAC1P'].value_counts()['1']/b['RAC1P'].count()
            
            p_Race2 = b['RAC1P'].value_counts()['2']/b['RAC1P'].count()
            
            p_Race3 = b['RAC1P'].value_counts()['3']/b['RAC1P'].count()
            
            p_Race4 = b['RAC1P'].value_counts()['4']/b['RAC1P'].count()
            
            p_Race5 = b['RAC1P'].value_counts()['5']/b['RAC1P'].count()
            
            p_Race6 = b['RAC1P'].value_counts()['6']/b['RAC1P'].count()
            
            p_Race7 = b['RAC1P'].value_counts()['7']/b['RAC1P'].count()
            
            p_Race8 = b['RAC1P'].value_counts()['8']/b['RAC1P'].count()
            
            p_Race9 = b['RAC1P'].value_counts()['9']/b['RAC1P'].count()
            
            pRace = [p_Race1, p_Race2, p_Race3, p_Race4, p_Race5, p_Race6, p_Race7, p_Race8, p_Race9]
            
            rng = np.random.default_rng()
            
            rnsClass = rng.multinomial(1, pClass,size=sum([row[1],row[2],row[3]]))
            
            Class.extend(rnsClass.argmax(axis=-1))
            
            rnsMilitary = rng.multinomial(1, pMilitary,size=sum([row[1],row[2],row[3]]))
            
            Military.extend(rnsMilitary.argmax(axis=-1))
            
            rnsEducation = rng.multinomial(1, pEducation,size=sum([row[1],row[2],row[3]]))
            
            Education.extend(rnsEducation.argmax(axis=-1))
            
            rnsRace = rng.multinomial(1, pRace,size=sum([row[1],row[2],row[3]]))
            
            Ethnicity.extend(rnsRace.argmax(axis=-1))
            
            rnsSex = rng.multinomial(1, pSex,size=sum([row[1],row[2],row[3]]))
            
            Gender.extend(rnsSex.argmax(axis=-1))
            
            for e in range(len(Earning)):
                if Earning[e]<0:
                    Earning[e]=0

            df['party']=party
            party_names = ['votes_D', 'votes_R', 'votes_oth']
            num_parties = 3
            encoder_party = dict(zip(party_names, range(num_parties)))
            df['party']=df['party'].map(lambda x: encoder_party[x])
            
            Gender=pd.get_dummies(Gender, prefix='Gender', prefix_sep='',dtype=np.int64)
            
            df['Gender0']=Gender['Gender0']
            
            df['Gender1']=Gender['Gender1']
            
            df['session']=region
            
            dfs['Class']=Class
            
            class_names = ['0','0','0','0','0','1','1','1','2']
            num_class = 9
            encoder_class = dict(zip(range(num_class), class_names))
            Class=dfs['Class'].map(lambda x: encoder_class[x])
            
            Class = pd.get_dummies(Class, prefix='Class', dtype=np.int64)
            
            if 'Class_0' in Class.columns:
                df['Class0'] = Class['Class_0']
            else:
                df['Class0'] = [0]*sum([row[1],row[2],row[3]])
            if 'Class_1' in Class.columns:
                df['Class1'] = Class['Class_1']
            else:
                df['Class1'] = [0]*sum([row[1],row[2],row[3]])
            if 'Class_2' in Class.columns:
                df['Class2'] = Class['Class_2']
            else:
                df['Class2'] = [0]*sum([row[1],row[2],row[3]])
            
            dfs['Ethnicity'] = Ethnicity
            
            eth_names = ['0','1','2','2','2','3','2','2','2']
            num_eth = 9
            encoder_eth = dict(zip(range(num_eth), eth_names))
            Ethnicity=dfs['Ethnicity'].map(lambda x: encoder_eth[x])
            
            Ethnicity=pd.get_dummies(Ethnicity, prefix='Ethnicity',dtype=np.int64)
            
            if 'Ethnicity_0' in Ethnicity.columns:
                df['Ethnicity0'] = Ethnicity['Ethnicity_0']
            else:
                df['Ethnicity0'] = [0]*sum([row[1],row[2],row[3]])
            if 'Ethnicity_1' in Ethnicity.columns:
                df['Ethnicity1'] = Ethnicity['Ethnicity_1']
            else:
                df['Ethnicity1'] = [0]*sum([row[1],row[2],row[3]])
            if 'Ethnicity_2' in Ethnicity.columns:
                df['Ethnicity2'] = Ethnicity['Ethnicity_2']
            else:
                df['Ethnicity2'] = [0]*sum([row[1],row[2],row[3]])
            if 'Ethnicity_3' in Ethnicity.columns:
                df['Ethnicity3'] = Ethnicity['Ethnicity_3']
            else:
                df['Ethnicity3'] = [0]*sum([row[1],row[2],row[3]])
            
            dfs['Age']=Age
            
            age_bins = [0,30,45,65,np.inf]
            age_names = [0,1,2,3]
            Age=pd.cut(dfs['Age'], bins=age_bins, labels=age_names)
            encoder_age = dict(zip([np.nan,0,1,2,3], [0,0,1,2,3]))
            Age=Age.map(lambda x: encoder_age[x]).astype(np.int64)
            
            Age = pd.get_dummies(Age, prefix='Age', dtype=np.int64)
            
            if 'Age_0' in Age.columns:
                df['Age0'] = Age['Age_0']
            else:
                df['Age0'] = [0]*sum([row[1],row[2],row[3]])
            if 'Age_1' in Age.columns:
                df['Age1'] = Age['Age_1']
            else:
                df['Age1'] = [0]*sum([row[1],row[2],row[3]])
            if 'Age_2' in Age.columns:
                df['Age2'] = Age['Age_2']
            else:
                df['Age2'] = [0]*sum([row[1],row[2],row[3]])
            if 'Age_3' in Age.columns:
                df['Age3'] = Age['Age_3']
            else:
                df['Age3'] = [0]*sum([row[1],row[2],row[3]])
            
            Earnings_bins = [-1,50000,100000,np.inf]
            earning_names = [0, 1, 2]
            Earning=pd.cut(Earning, bins=Earnings_bins, labels=earning_names)
            encoder_earning = dict(zip(earning_names, range(3)))
            Earning=Earning.map(lambda x: encoder_earning[x]).astype(np.int64)
            
            Earning = pd.get_dummies(Earning, prefix='Earning', dtype=np.int64)
            
            if 'Earning_0' in Earning.columns:
                df['Earning0'] = Earning['Earning_0']
            else:
                df['Earning0'] = [0]*sum([row[1],row[2],row[3]])
            if 'Earning_1' in Earning.columns:
                df['Earning1'] = Earning['Earning_1']
            else:
                df['Earning1'] = [0]*sum([row[1],row[2],row[3]])
            if 'Earning_2' in Earning.columns:
                df['Earning2'] = Earning['Earning_2']
            else:
                df['Earning2'] = [0]*sum([row[1],row[2],row[3]])
            
            dfs['Military']=Military
            
            mil_names = ['0','0','0','1',]
            num_mil = 4
            encoder_mil = dict(zip(range(num_mil), mil_names))
            Military=dfs['Military'].map(lambda x: encoder_mil[x])
            
            Military = pd.get_dummies(Military, prefix='Military',dtype=np.int64)
            
            if 'Military_0' in Military.columns:
                df['Military0'] = Military['Military_0']
            else:
                df['Military0'] = [0]*sum([row[1],row[2],row[3]])
            if 'Military_1' in Military.columns:
                df['Military1'] = Military['Military_1']
            else:
                df['Military1'] = [0]*sum([row[1],row[2],row[3]])

            dfs['Education']=Education
            
            edu_names = ['0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','1','1','1','1','1','2','2']
            num_edu = 24
            encoder_edu = dict(zip(range(num_edu), edu_names))
            Education=dfs['Education'].map(lambda x: encoder_edu[x])
            
            Education = pd.get_dummies(Education, prefix='Education', dtype=np.int64)
            
            if 'Education_0' in Education.columns:
                df['Education0'] = Education['Education_0']
            else:
                df['Education0'] = [0]*sum([row[1],row[2],row[3]])
            if 'Education_1' in Education.columns:
                df['Education1'] = Education['Education_1']
            else:
                df['Education1'] = [0]*sum([row[1],row[2],row[3]])
            if 'Education_2' in Education.columns:
                df['Education2'] = Education['Education_2']
            else:
                df['Education2'] = [0]*sum([row[1],row[2],row[3]])
                
            df['choice']= [1]*sum([row[1],row[2],row[3]])
            
            count2 = count + sum([row[1],row[2],row[3]])
            
            df['user']=range(count, count2)
            
            count = sum([row[1],row[2],row[3]])
            
            df.to_csv('folder/votes.csv', index=False)
            
            new_name = 'folder/'+Filenames[i]+'.csv' 
            
            os.rename('folder/votes.csv', new_name)
            
    i+=1

