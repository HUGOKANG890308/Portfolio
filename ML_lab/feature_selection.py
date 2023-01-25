from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier
import xgbfir
from sklearn.utils import shuffle
#dataset=load_dataset.load_data()
class feature_selction():
    def __init__(self,dataset,seed,year,max_feature_size,target):
        '''        
        :dataset= data which you want, type=DataFrame
        :seed,random seed
        :year  the last training year
        :max_feature , how many feature
        :return: feature selection result
        :rtype: DataFrame
        '''
        self.r=seed
        self.x1=year
        self.dataset=dataset
        self.count=max_feature_size
        self.target=target
    def RF_feature_selction(self):
        trainingdata= self.dataset[self.dataset['fyear'] <= self.x1] 
        x=trainingdata.copy().drop(['gvkey','fyear',self.target],axis = 1)
        y=trainingdata[self.target]
        rf = RandomForestClassifier(random_state=self.r)
        rf = rf.fit(x,y)
        feature_names= x.keys().tolist()        
        df_ipt = pd.DataFrame({'feature':feature_names,  
                       'feature_importance':rf.feature_importances_.tolist() })
        df_ipt=df_ipt.sort_values(by='feature_importance', ascending=False)
        df_ipt=df_ipt.iloc[:self.count]
        feature=df_ipt['feature'].tolist()
        newdf=pd.DataFrame()
        for i in feature:
            newdf[i]= self.dataset[i]
        RF_df=newdf
        RF_df['gvkey']=self.dataset['gvkey']
        RF_df['fyear']=self.dataset['fyear']
        RF_df[self.target]=self.dataset[self.target]
    
        return RF_df
    
    
    def lasso_feature_selction(self):
        trainingdata= self.dataset[self.dataset['fyear'] <= self.x1] 
        x=trainingdata.copy().drop(['gvkey','fyear',self.target],axis = 1)
        y=trainingdata[self.target]
        reg=LassoCV(cv=self.count, random_state=self.r)
        reg=reg.fit(x,y)
        coef=pd.Series(reg.coef_,index=x.columns)
        
        coef=pd.DataFrame(coef)
        coef=coef.drop(coef[(coef[0])==0].index)
        coef=coef.apply(lambda x:abs(x))
        coef=coef.sort_values(by=0,ascending=False)
        coef=coef.iloc[:self.count]
    
        feature=coef.index.tolist()
        newdf=pd.DataFrame()
        for i in feature:
            newdf[i]= self.dataset[i]
        lasso_df= newdf
        lasso_df['gvkey']=self.dataset['gvkey']
        lasso_df['fyear']=self.dataset['fyear']
        lasso_df[self.target]=self.dataset[self.target]    
        return lasso_df
    
    
    def XGB_feature_selction(self):
        trainingdata= self.dataset[self.dataset['fyear'] <= self.x1] 
        x=trainingdata.copy().drop(['gvkey','fyear',self.target],axis = 1)
        y=trainingdata[self.target]
        XGB = XGBClassifier(random_state=self.r)
        XGB  = XGB .fit(x,y)
        feature_names= x.keys().tolist()  
        xgbfir.saveXgbFI(XGB, feature_names=feature_names, OutputXlsxFile='KLD ML\\testxgb.xlsx') 
        df_ipt=pd.read_excel('KLD ML\\testxgb.xlsx',sheet_name='Interaction Depth 0')    
        df_ipt=df_ipt.iloc[:self.count]
        feature=df_ipt['Interaction'].tolist()
        newdf=pd.DataFrame()
        for i in feature:
            newdf[i]= self.dataset[i]
        XGB_df=newdf
        XGB_df['gvkey']=self.dataset['gvkey']
        XGB_df['fyear']=self.dataset['fyear']
        XGB_df[self.target]=self.dataset[self.target] 
        return XGB_df
    def __getitem__(self,method):
        '''        
        :param method: 'XGB','RF','Lasso','No'
        :type method: ['param']
        :return: feature selection result
        :rtype: DataFrame
        '''
        if method=='XGB':
            output=self.XGB_feature_selction()
            return output
        elif method=='RF':
            output=self.RF_feature_selction()
            return output
        elif method=='Lasso':
            output=self.lasso_feature_selction()
            return output
        elif method=='No':
            output=self.dataset
            return output
        error_msg = "Please input right param :param method: 'XGB','RF','Lasso','No'"
        return error_msg
class train_test_selection():
    def __init__(self,dataset,year,n_years):
        '''        
        ;dataset= data which you want, type=DataFrame
        ;seed,random seed
        ;year  the last training year
        ;n_years how many years in test     
        ;rtype: DataFrame
        '''
        self.dataset=dataset
        self.year=year
        self.n_years=n_years
        self.trainingdata=self.dataset[self.dataset['fyear']<=self.year]    
        if self.n_years==1:
                self.test_dataset=self.dataset[(self.dataset['fyear'] == self.year+1)] 
                len1=len(self.test_dataset)
                self.test_size,self.vaild_size=int(0.8*len1),int(0.2*len1)
                self.test_dataset=shuffle(self.test_dataset)
                self.vaildingdata=self.test_dataset.iloc[:self.vaild_size]
                self.testingdata= self.test_dataset.iloc[self.vaild_size:self.test_size]
        if self.n_years==2:
            if 2018-self.year>=2:
                self.test_dataset= self.dataset[(self.dataset['fyear'] == self.year+1)| (self.dataset['fyear'] == self.year+2)] 
                len1=len(self.test_dataset)
                self.test_size,self.vaild_size=int(0.8*len1),int(0.2*len1)
                self.test_dataset=shuffle(self.test_dataset)
                self.vaildingdata=self.test_dataset.iloc[:self.vaild_size]
                self.testingdata= self.test_dataset.iloc[self.vaild_size:self.test_size]
            else : return('error,Not enough time left')
        if self.n_years==3:
            if 2018-self.year>=3:
                self.test_dataset= self.dataset[(self.dataset['fyear'] == self.year+1)|
                 (self.dataset['fyear'] == self.year+2)| (self.dataset['fyear'] == self.year+3)] 
                len1=len(self.test_dataset)
                self.test_size,self.vaild_size=int(0.8*len1),int(0.2*len1)
                self.test_dataset=shuffle(self.test_dataset)
                self.vaildingdata=self.test_dataset.iloc[:self.vaild_size]
                self.testingdata= self.test_dataset.iloc[self.vaild_size:self.test_size]
            else : return('error,Not enough time left')
        if self.n_years==4:
            if 2018-self.year>=4:
                self.test_dataset= self.dataset[(self.dataset['fyear'] == self.year+1)| 
                (self.dataset['fyear'] == self.year+2)| (self.dataset['fyear'] == self.year+3)|
                (self.dataset['fyear'] == self.year+4)]
                len1=len(self.test_dataset)
                self.test_size,self.vaild_size=int(0.8*len1),int(0.2*len1)
                self.test_dataset=shuffle(self.test_dataset)
                self.vaildingdata=self.test_dataset.iloc[:self.vaild_size]
                self.testingdata= self.test_dataset.iloc[self.vaild_size:self.test_size]
            else : return('error,Not enough time left')
    def __getitem__(self,param):
        '''        
        :param method: ['train','vaild','test']
        :type method: param
        :return: trainingdata, if train='train'
                 vaildingdata,if train='vaild'
                 testingdata, if train='test'
        :rtype: DataFrame
        '''
        if param=='train':
            return  self.trainingdata
        elif param=='vaild':
            return self.vaildingdata
        elif param=='test':
            return self.testingdata
