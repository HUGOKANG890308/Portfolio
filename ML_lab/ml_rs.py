#ML model random_split train test
from ML_lab.feature_selction import feature_selction
from ML_lab.feature_selction  import train_test_selection
from ML_lab.load_dataset import load_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from imblearn.metrics import sensitivity_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import os
os.chdir("C:\\Users\\User\\OneDrive\\桌面Dell\\1008 meet")
class metric():
    def __init__(self) :
        '''
        metric1.model=[]
        metric1.f1_1=[]
        metric1.f1_0=[]
        metric1.precision_0=[]
        metric1.precision_1=[]
        metric1.recall_0=[]
        metric1.recall_1=[]
        metric1.accuracy=[]
        metric1.model_name=[]
        metric1.TP=[]
        metric1.FP=[]
        metric1.FN=[]
        metric1.TN=[]
        metric1.seed=[]
        metric1.TP_rate=[]
        metric1.FN_rate=[]
        metric1.TN_rate=[]
        metric1.FP_rate=[]
        metric1.year=[]
        metric1.roc0=[]
        metric1.roc1=[]
        '''
        self.model=[]
        self.f1_1=[]
        self.f1_0=[]
        self.precision_0=[]
        self.precision_1=[]
        self.recall_0=[]
        self.recall_1=[]
        self.accuracy=[]
        self.model_name=[]
        self.TP=[]
        self.FP=[]
        self.FN=[]
        self.TN=[]
        self.seed=[]
        self.TP_rate=[]
        self.FN_rate=[]
        self.TN_rate=[]
        self.FP_rate=[]
    
        self.roc0=[]
        self.roc1=[]
metric1=metric()

class Logestic():
    def __init__(self,dataset,seed,target,select,test_size):
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :select feature selection in what method ['XGB','RF','Lasso'.'No']
       :n_years how many years in test     
       '''
       self.select=select
       self.target=target
       self.dataset=dataset
       self.r=seed
       self.test_size=test_size
       X=self.dataset.drop(['gvkey','fyear',self.target],axis = 1)
       Y=self.dataset[self.target]
       self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=self.test_size)
    def ML_model(self,ml_model):
        ml_model=ml_model.fit(self.x_train, self.y_train)
        metric1.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        metric1.roc0.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 0], average=None))
        metric1.roc1.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 1], average=None))
        self.y_pred= ml_model.predict(self.x_test)
        print(metrics.classification_report(self.y_test,self.y_pred))
        print('f1_score',f1_score(self.y_test, self.y_pred) )
        metric1.model_name.append(self.name)
        self.score(self.y_pred)
        print('random_state=',self.r)
        print('y_test.shape',self.y_test.shape)
        try:
            os.remove('KLD ML\\testxgb1.xlsx')
        except:pass
    def score(self,y_pred):
        print('y test',self.y_test.shape) 
        metric1.f1_0.append(f1_score(self.y_test,self.y_pred,average=None)[0])
        metric1.f1_1.append(f1_score(self.y_test,y_pred,average=None)[1])
        metric1.precision_0.append(precision_score(self.y_test, self.y_pred,average=None)[0])
        metric1.precision_1.append(precision_score(self.y_test,self.y_pred,average=None)[1])
        metric1.recall_0.append(sensitivity_score(self.y_test,self.y_pred,average=None)[0])
        metric1.recall_1.append(sensitivity_score(self.y_test,self.y_pred,average=None)[1])
        metric1.accuracy.append(accuracy_score(self.y_test, self.y_pred ))    
        metric1.TP.append(confusion_matrix(self.y_test,self.y_pred)[0][0])
        metric1.FP.append(confusion_matrix(self.y_test,self.y_pred)[1][0])
        metric1.TN.append(confusion_matrix(self.y_test,self.y_pred)[1][1])
        metric1.FN.append(confusion_matrix(self.y_test,self.y_pred)[0][1])
        metric1.TP_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][0]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.FN_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][1]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.TN_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][1]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.FP_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][0]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.seed.append(self.r)
    def logistic_basic(self):
        log=LogisticRegression(random_state=self.r)
        self.name='logestic_basic_'+self.select+'_select'
        
        print(self.name)
        self.ML_model(log)
    def logistic_EasyEnsemble(self,n):
        log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         LogisticRegression(random_state=self.r))
        self.name='logestic_EasyEnsemble_'+self.select+'_select'

        print(self.name)
        self.ML_model(log)
    def logistic_BalancedBagging(self,n):
        log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         LogisticRegression(random_state=self.r))
        self.name='logestic_BalancedBagging_'+self.select+'_select'
        
        print(self.name)
        self.ML_model(log)
class SVM():
    def __init__(self,dataset,seed,target,select,test_size):
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :select feature selection in what method ['XGB','RF','Lasso'.'No']
       :n_years how many years in test     
       '''
       self.select=select
       self.target=target
       self.dataset=dataset
       self.r=seed
       self.test_size=test_size
       X=self.dataset.drop(['gvkey','fyear',self.target],axis = 1)
       Y=self.dataset[self.target]
       self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=self.test_size)
    def score(self,y_pred):
        print('y test',self.y_test.shape)  
        metric1.f1_0.append(f1_score(self.y_test,self.y_pred,average=None)[0])
        metric1.f1_1.append(f1_score(self.y_test,y_pred,average=None)[1])
        metric1.precision_0.append(precision_score(self.y_test, self.y_pred,average=None)[0])
        metric1.precision_1.append(precision_score(self.y_test,self.y_pred,average=None)[1])
        metric1.recall_0.append(sensitivity_score(self.y_test,self.y_pred,average=None)[0])
        metric1.recall_1.append(sensitivity_score(self.y_test,self.y_pred,average=None)[1])
        metric1.accuracy.append(accuracy_score(self.y_test, self.y_pred ))    
        metric1.TP.append(confusion_matrix(self.y_test,self.y_pred)[0][0])
        metric1.FP.append(confusion_matrix(self.y_test,self.y_pred)[1][0])
        metric1.TN.append(confusion_matrix(self.y_test,self.y_pred)[1][1])
        metric1.FN.append(confusion_matrix(self.y_test,self.y_pred)[0][1])
        metric1.TP_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][0]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.FN_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][1]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.TN_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][1]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.FP_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][0]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.seed.append(self.r)

    def ML_model(self,ml_model):
        ml_model=ml_model.fit(self.x_train, self.y_train)
        metric1.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        metric1.roc0.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 0], average=None))
        metric1.roc1.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 1], average=None))
        self.y_pred= ml_model.predict(self.x_test)
        print(metrics.classification_report(self.y_test,self.y_pred))
        print('f1_score',f1_score(self.y_test, self.y_pred) )
        metric1.model_name.append(self.name)
        self.score(self.y_pred)
        print('random_state=',self.r)
        print('y_test.shape',self.y_test.shape)
        print('y_years=',self.years)
        try:
            os.remove('KLD ML\\testxgb1.xlsx')
        except:pass
    def SVM_basic(self):
        log=SVC(kernel='rbf',random_state =self.r)
        self.name='SVM_basic_'+self.select+'_select'
        
        print(self.name)
        self.ML_model(log)
    def SVM_EasyEnsemble(self,n):
        log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         SVC(kernel='rbf',random_state =self.r))
        self.name='SVM_EasyEnsemble_'+self.select+'_select'

        print(self.name)
        self.ML_model(log)
    def SVM_BalancedBagging(self,n):
        log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         SVC(kernel='rbf',random_state =self.r))
        self.name='SVM_BalancedBagging_'+self.select+'_select'
        
        print(self.name)
        self.ML_model(log)

    
class Random_Forest():
    def __init__(self,dataset,seed,target,select,test_size):
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :select feature selection in what method ['XGB','RF','Lasso'.'No']
       :n_years how many years in test     
       '''
       self.select=select
       self.target=target
       self.dataset=dataset
       self.r=seed
       self.test_size=test_size
       X=self.dataset.drop(['gvkey','fyear',self.target],axis = 1)
       Y=self.dataset[self.target]
       self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=self.test_size)
    def score(self,y_pred):
        print('y test',self.y_test.shape)    
        metric1.f1_0.append(f1_score(self.y_test,self.y_pred,average=None)[0])
        metric1.f1_1.append(f1_score(self.y_test,y_pred,average=None)[1])
        metric1.precision_0.append(precision_score(self.y_test, self.y_pred,average=None)[0])
        metric1.precision_1.append(precision_score(self.y_test,self.y_pred,average=None)[1])
        metric1.recall_0.append(sensitivity_score(self.y_test,self.y_pred,average=None)[0])
        metric1.recall_1.append(sensitivity_score(self.y_test,self.y_pred,average=None)[1])
        metric1.accuracy.append(accuracy_score(self.y_test, self.y_pred ))    
        metric1.TP.append(confusion_matrix(self.y_test,self.y_pred)[0][0])
        metric1.FP.append(confusion_matrix(self.y_test,self.y_pred)[1][0])
        metric1.TN.append(confusion_matrix(self.y_test,self.y_pred)[1][1])
        metric1.FN.append(confusion_matrix(self.y_test,self.y_pred)[0][1])
        metric1.TP_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][0]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.FN_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][1]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.TN_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][1]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.FP_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][0]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.seed.append(self.r)

    def ML_model(self,ml_model):
        ml_model=ml_model.fit(self.x_train, self.y_train)
        metric1.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        metric1.roc0.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 0], average=None))
        metric1.roc1.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 1], average=None))
        self.y_pred= ml_model.predict(self.x_test)
        print(metrics.classification_report(self.y_test,self.y_pred))
        print('f1_score',f1_score(self.y_test, self.y_pred) )
        metric1.model_name.append(self.name)
        self.score(self.y_pred)
        print('random_state=',self.r)
        print('y_test.shape',self.y_test.shape)
        print('y_years=',self.years)
        try:
            os.remove('KLD ML\\testxgb1.xlsx')
        except:pass
    def RF_basic(self):
        log=RandomForestClassifier(random_state=self.r)
        self.name='RF_basic_'+self.select+'_select'  
        print(self.name)
        self.ML_model(log)
    def RF_EasyEnsemble(self,n):
        log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                        RandomForestClassifier(random_state=self.r))
        self.name='RF_EasyEnsemble_'+self.select+'_select'
        print(self.name)
        self.ML_model(log)
    def RF_BalancedBagging(self,n):
        log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         RandomForestClassifier(random_state=self.r))
        self.name='RF_BalancedBagging_'+self.select+'_select'
        print(self.name)
        self.ML_model(log)

class XGBOOST():
    def __init__(self,dataset,seed,target,select,test_size):
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target 
       :select feature selection in what method ['XGB','RF','Lasso'.'No']
       :n_years how many years in test     
       '''
       self.select=select
       self.target=target
       self.dataset=dataset
       self.r=seed
       self.test_size=test_size
       X=self.dataset.drop(['gvkey','fyear',self.target],axis = 1)
       Y=self.dataset[self.target]
       self.x_train,self.x_test,self.y_train,self.y_test=train_test_split(X,Y,test_size=self.test_size)
    def score(self,y_pred):
        print('y test',self.y_test.shape)    
        metric1.f1_0.append(f1_score(self.y_test,self.y_pred,average=None)[0])
        metric1.f1_1.append(f1_score(self.y_test,y_pred,average=None)[1])
        metric1.precision_0.append(precision_score(self.y_test, self.y_pred,average=None)[0])
        metric1.precision_1.append(precision_score(self.y_test,self.y_pred,average=None)[1])
        metric1.recall_0.append(sensitivity_score(self.y_test,self.y_pred,average=None)[0])
        metric1.recall_1.append(sensitivity_score(self.y_test,self.y_pred,average=None)[1])
        metric1.accuracy.append(accuracy_score(self.y_test, self.y_pred ))    
        metric1.TP.append(confusion_matrix(self.y_test,self.y_pred)[0][0])
        metric1.FP.append(confusion_matrix(self.y_test,self.y_pred)[1][0])
        metric1.TN.append(confusion_matrix(self.y_test,self.y_pred)[1][1])
        metric1.FN.append(confusion_matrix(self.y_test,self.y_pred)[0][1])
        metric1.TP_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][0]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.FN_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][1]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        metric1.TN_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][1]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.FP_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][0]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        metric1.seed.append(self.r)

    def ML_model(self,ml_model):
        ml_model=ml_model.fit(self.x_train, self.y_train)
        metric1.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        metric1.roc0.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 0], average=None))
        metric1.roc1.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 1], average=None))
        self.y_pred= ml_model.predict(self.x_test)
        print(metrics.classification_report(self.y_test,self.y_pred))
        print('f1_score',f1_score(self.y_test, self.y_pred) )
        metric1.model_name.append(self.name)
        self.score(self.y_pred)
        print('random_state=',self.r)
        print('y_test.shape',self.y_test.shape)
        print('y_years=',self.years)
        try:
            os.remove('KLD ML\\testxgb1.xlsx')
        except:pass
    def XGB_basic(self):
        try:
            log=XGBClassifier(random_state=self.r,tree_method='gpu_hist') 
            print(self.name)
            self.ML_model(log)           
        except: 
            log=XGBClassifier(random_state=self.r)
        self.name='XGB_basic_'+self.select+'_select'  
        print(self.name)
        self.ML_model(log)
    def XGB_EasyEnsemble(self,n):
        try:
            log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                        XGBClassifier(random_state=self.r,tree_method='gpu_hist') )
            print(self.name)
            self.ML_model(log)
        except:     
            log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                        XGBClassifier(random_state=self.r) )
            
        self.name='XGB_EasyEnsemble_'+self.select+'_select'
        print(self.name)
        self.ML_model(log)
    def XGB_BalancedBagging(self,n):
        try:
            log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         XGBClassifier(random_state=self.r,tree_method='gpu_hist'))
            print(self.name)
            self.ML_model(log)
        except:
            log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,base_estimator = 
                                         XGBClassifier(random_state=self.r))
        self.name='XGB_BalancedBagging_'+self.select+'_select'
        print(self.name)
        self.ML_model(log)
 
'''
dataset=load_data('FIN+KLD')
model_log=Logestic(dataset,seed=0,year=2017,target='Brupt_crsp_t1',select='RF',n_years=1)
model_log.logistic_basic()  
model_log.logistic_BalancedBagging(10) 
model_log.logistic_EasyEnsemble(10)
'''    
