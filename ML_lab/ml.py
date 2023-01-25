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
import os
#os.chdir("C:\\Users\\User\\OneDrive\\桌面Dell\\1008 meet")
class metric():
    def __init__(self) :
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
        self.year=[]
        self.roc0=[]
        self.roc1=[]
test_metric=metric()
vaild_metric=metric()
class basic:
    def ML_model(self,ml_model):
        ml_model=ml_model.fit(self.x_train, self.y_train)
        test_metric.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        test_metric.roc0.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 0], average=None))
        test_metric.roc1.append(roc_auc_score(self.y_test,  ml_model .predict_proba(self.x_test)[:, 1], average=None))
        self.y_pred= ml_model.predict(self.x_test)
        print('test_result')
        print(metrics.classification_report(self.y_test,self.y_pred))
        print('f1_score',f1_score(self.y_test, self.y_pred) )
        test_metric.model_name.append(self.name)
        self.test_score(self.y_pred)
        print('random_state=',self.r)
        print('y_test.shape',self.y_test.shape)
        print('y_years=',self.years)
        
        print('vaild_result')
        vaild_metric.model.append(ml_model)
        #若隨機抽取一個陽性樣本和一個陰性樣本，分類器正確判斷陽性樣本的值高於陰性樣本之機率 {\displaystyle =AUC}{\displaystyle =AUC}[1]。
        #numpy array [:,1]所有rows 取第一個cols
        vaild_metric.roc0.append(roc_auc_score(self.y_vaild,  ml_model .predict_proba(self.x_vaild)[:, 0], average=None))
        vaild_metric.roc1.append(roc_auc_score(self.y_vaild,  ml_model .predict_proba(self.x_vaild)[:, 1], average=None))
        self.y_pred_vaild= ml_model.predict(self.x_vaild)
    
        print(metrics.classification_report(self.y_vaild,self.y_pred_vaild))
        print('f1_score',f1_score(self.y_vaild, self.y_pred_vaild) )
        vaild_metric.model_name.append(self.name)
        self.vaild_score(self.y_pred_vaild)
        print('random_state=',self.r)
        print('y_test.shape',self.y_vaild.shape)
        print('y_years=',self.years)

        try:
            os.remove('KLD ML\\testxgb1.xlsx')
        except:pass
    def test_score(self,y_pred):
        print('y test',self.y_test.shape)
        if self.n_years==1:
            test_metric.year.append(self.years+1)
        if self.n_years==2:
            period=str(self.years+1)+'~'+str(self.years+2) 
            test_metric.year.append(period)
        if self.n_years==3:
            period=str(self.years+1)+'~'+str(self.years+3) 
            test_metric.year.append(period)
        if self.n_years==4:
            period=str(self.years+1)+'~'+str(self.years+4) 
            test_metric.year.append(period)    
        test_metric.f1_0.append(f1_score(self.y_test,self.y_pred,average=None)[0])
        test_metric.f1_1.append(f1_score(self.y_test,y_pred,average=None)[1])
        test_metric.precision_0.append(precision_score(self.y_test, self.y_pred,average=None)[0])
        test_metric.precision_1.append(precision_score(self.y_test,self.y_pred,average=None)[1])
        test_metric.recall_0.append(sensitivity_score(self.y_test,self.y_pred,average=None)[0])
        test_metric.recall_1.append(sensitivity_score(self.y_test,self.y_pred,average=None)[1])
        test_metric.accuracy.append(accuracy_score(self.y_test, self.y_pred ))    
        test_metric.TP.append(confusion_matrix(self.y_test,self.y_pred)[0][0])
        test_metric.FP.append(confusion_matrix(self.y_test,self.y_pred)[1][0])
        test_metric.TN.append(confusion_matrix(self.y_test,self.y_pred)[1][1])
        test_metric.FN.append(confusion_matrix(self.y_test,self.y_pred)[0][1])
        test_metric.TP_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][0]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        test_metric.FN_rate.append(confusion_matrix(self.y_test,self.y_pred)[0][1]/(confusion_matrix(self.y_test,self.y_pred)[0][0]+confusion_matrix(self.y_test,self.y_pred)[0][1]))
        test_metric.TN_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][1]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        test_metric.FP_rate.append(confusion_matrix(self.y_test,self.y_pred)[1][0]/(confusion_matrix(self.y_test,self.y_pred)[1][1]+confusion_matrix(self.y_test,self.y_pred)[1][0]))
        test_metric.seed.append(self.r)
    def vaild_score(self,y_pred_vaild):
        print('y test',self.y_test.shape)
        if self.n_years==1:
            vaild_metric.year.append(self.years+1)
        if self.n_years==2:
            period=str(self.years+1)+'~'+str(self.years+2) 
            vaild_metric.year.append(period)
        if self.n_years==3:
            period=str(self.years+1)+'~'+str(self.years+3) 
            vaild_metric.year.append(period)
        if self.n_years==4:
            period=str(self.years+1)+'~'+str(self.years+4) 
            vaild_metric.year.append(period)    
        vaild_metric.f1_0.append(f1_score(self.y_vaild,self.y_pred_vaild,average=None)[0])
        vaild_metric.f1_1.append(f1_score(self.y_vaild,y_pred_vaild,average=None)[1])
        vaild_metric.precision_0.append(precision_score(self.y_vaild, self.y_pred_vaild,average=None)[0])
        vaild_metric.precision_1.append(precision_score(self.y_vaild,self.y_pred_vaild,average=None)[1])
        vaild_metric.recall_0.append(sensitivity_score(self.y_vaild,self.y_pred_vaild,average=None)[0])
        vaild_metric.recall_1.append(sensitivity_score(self.y_vaild,self.y_pred_vaild,average=None)[1])
        vaild_metric.accuracy.append(accuracy_score(self.y_vaild, self.y_pred_vaild ))    
        vaild_metric.TP.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][0])
        vaild_metric.FP.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][0])
        vaild_metric.TN.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][1])
        vaild_metric.FN.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][1])
        vaild_metric.TP_rate.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][0]/(confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][0]+confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][1]))
        vaild_metric.FN_rate.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][1]/(confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][0]+confusion_matrix(self.y_vaild,self.y_pred_vaild)[0][1]))
        vaild_metric.TN_rate.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][1]/(confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][1]+confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][0]))
        vaild_metric.FP_rate.append(confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][0]/(confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][1]+confusion_matrix(self.y_vaild,self.y_pred_vaild)[1][0]))
        vaild_metric.seed.append(self.r)
class Logestic(basic):
    def __init__(self,dataset,seed,year,target,select,n_years):
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
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)
       df=feature_selction(self.dataset, self.r,self.years,20,target=self.target)[self.select]
       '''
       (dataset: Any, seed: Any, year: Any, max_feature_size: Any, target: Any) -> None
       :dataset= data which you want, type=DataFrame
        :seed,random seed
        :year the last training year
        :select , ['XGB','RF','Lasso'.'No']
        :return: feature selection result
        :rtype: DataFrame     
       '''
       self.data=train_test_selection(dataset=df,year=self.years,n_years=self.n_years)
       self.training_data = self.data.__getitem__('train')
       self.testing_data=self.data.__getitem__('test') 
       self.vailding_data=self.data.__getitem__('vaild') 
       self.x_train,self.x_test=self.training_data.drop([self.target,'gvkey','fyear'],axis=1),self.testing_data.drop([self.target,'gvkey','fyear'],axis=1)
       self.y_train,self.y_test=self.training_data[self.target],self.testing_data[self.target]
       self.x_vaild,self.y_vaild=self.vailding_data.drop([self.target,'gvkey','fyear'],axis=1),self.vailding_data[self.target]
    def logistic_basic(self):
        log=LogisticRegression(random_state=self.r)
        self.name='logestic_basic_'+self.select+'_select'
        
        print(self.name)
        super().ML_model(log)
    def logistic_EasyEnsemble(self,n):
        log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         LogisticRegression(random_state=self.r))
        self.name='logestic_EasyEnsemble_'+self.select+'_select'

        print(self.name)
        super().ML_model(log)
    def logistic_BalancedBagging(self,n):
        log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         LogisticRegression(random_state=self.r))
        self.name='logestic_BalancedBagging_'+self.select+'_select'
        
        print(self.name)
        super().ML_model(log)

class SVM(basic):
    def __init__(self,dataset,seed,year,target,select,n_years):
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target ,type str
       :select feature selection in what method ['XGB','RF','Lasso'.'No']
       :n_years how many years in test     
       '''
       self.select=select
       self.target=target
       self.dataset=dataset
       self.r=seed
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)
       df=feature_selction(self.dataset, self.r,self.years,20,target=self.target)[self.select]
       '''
       (dataset: Any, seed: Any, year: Any, max_feature_size: Any, target: Any) -> None
       :dataset= data which you want, type=DataFrame
        :seed,random seed
        :year the last training year
        :select , ['XGB','RF','Lasso'.'No']
        :return: feature selection result
        :rtype: DataFrame     
       '''
    
       self.data=train_test_selection(dataset=df,year=self.years,n_years=self.n_years)
       self.training_data = self.data.__getitem__('train')
       self.testing_data=self.data.__getitem__('test') 
       self.vailding_data=self.data.__getitem__('vaild') 
       self.x_train,self.x_test=self.training_data.drop([self.target,'gvkey','fyear'],axis=1),self.testing_data.drop([self.target,'gvkey','fyear'],axis=1)
       self.y_train,self.y_test=self.training_data[self.target],self.testing_data[self.target]
       self.x_vaild,self.y_vaild=self.vailding_data.drop([self.target,'gvkey','fyear'],axis=1),self.vailding_data[self.target]
    def SVM_basic(self):
        log=SVC(kernel='rbf',random_state =self.r)
        self.name='SVM_basic_'+self.select+'_select'
        
        print(self.name)
        super().ML_model(log)
    def SVM_EasyEnsemble(self,n):
        log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         SVC(kernel='rbf',random_state =self.r))
        self.name='SVM_EasyEnsemble_'+self.select+'_select'

        print(self.name)
        super().ML_model(log)
    def SVM_BalancedBagging(self,n):
        log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         SVC(kernel='rbf',random_state =self.r))
        self.name='SVM_BalancedBagging_'+self.select+'_select'
        
        print(self.name)
        super().ML_model(log)

    
class Random_Forest():
    def __init__(self,dataset,seed,year,target,select,n_years):
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
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)
       df=feature_selction(self.dataset, self.r,self.years,20,target=self.target)[self.select]
       '''
       (dataset: Any, seed: Any, year: Any, max_feature_size: Any, target: Any) -> None
       :dataset= data which you want, type=DataFrame
        :seed,random seed
        :year the last training year
        :select , ['XGB','RF','Lasso'.'No']
        :return: feature selection result
        :rtype: DataFrame     
       '''
       self.data=train_test_selection(dataset=df,year=self.years,n_years=self.n_years)
       self.training_data = self.data.__getitem__('train')
       self.testing_data=self.data.__getitem__('test') 
       self.vailding_data=self.data.__getitem__('vaild') 
       self.x_train,self.x_test=self.training_data.drop([self.target,'gvkey','fyear'],axis=1),self.testing_data.drop([self.target,'gvkey','fyear'],axis=1)
       self.y_train,self.y_test=self.training_data[self.target],self.testing_data[self.target]
       self.x_vaild,self.y_vaild=self.vailding_data.drop([self.target,'gvkey','fyear'],axis=1),self.vailding_data[self.target]

    def RF_basic(self):
        log=RandomForestClassifier(random_state=self.r)
        self.name='RF_basic_'+self.select+'_select'  
        print(self.name)
        super().ML_model(log)
    def RF_EasyEnsemble(self,n):
        log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                        RandomForestClassifier(random_state=self.r))
        self.name='RF_EasyEnsemble_'+self.select+'_select'
        print(self.name)
        super().ML_model(log)
    def RF_BalancedBagging(self,n):
        log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         RandomForestClassifier(random_state=self.r))
        self.name='RF_BalancedBagging_'+self.select+'_select'
        print(self.name)
        super().ML_model(log)

class XGBOOST():
    def __init__(self,dataset,seed,year,target,select,n_years):
       '''
       :dataset= data which you want, type=DataFrame
       :seed,random seed
       :year the last training year
       :target y target ,type str
       :select feature selection in what method ['XGB','RF','Lasso'.'No']
       :n_years how many years in test     
       '''
       self.select=select
       self.target=target
       self.dataset=dataset
       self.r=seed
       self.n_years=n_years
       self.years=year
       #feature_selction(self.dataset, self.r,self.years,20,target=self.target)
       df=feature_selction(self.dataset, self.r,self.years,20,target=self.target)[self.select]
       '''
       (dataset: Any, seed: Any, year: Any, max_feature_size: Any, target: Any) -> None
       :dataset= data which you want, type=DataFrame
        :seed,random seed
        :year the last training year
        :select , ['XGB','RF','Lasso'.'No']
        :return: feature selection result
        :rtype: DataFrame     
       '''
       self.data=train_test_selection(dataset=df,year=self.years,n_years=self.n_years)
       self.training_data = self.data.__getitem__('train')
       self.testing_data=self.data.__getitem__('test') 
       self.vailding_data=self.data.__getitem__('vaild') 
       self.x_train,self.x_test=self.training_data.drop([self.target,'gvkey','fyear'],axis=1),self.testing_data.drop([self.target,'gvkey','fyear'],axis=1)
       self.y_train,self.y_test=self.training_data[self.target],self.testing_data[self.target]
       self.x_vaild,self.y_vaild=self.vailding_data.drop([self.target,'gvkey','fyear'],axis=1),self.vailding_data[self.target]
    def XGB_basic(self):
        try:
            log=XGBClassifier(random_state=self.r,tree_method='gpu_hist') 
            print(self.name)
            self.ML_model(log)           
        except: 
            log=XGBClassifier(random_state=self.r)
        self.name='XGB_basic_'+self.select+'_select'  
        print(self.name)
        super().ML_model(log)
    def XGB_EasyEnsemble(self,n):
        try:
            log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                        XGBClassifier(random_state=self.r,tree_method='gpu_hist') )
            print(self.name)
            self.ML_model(log)
        except:     
            log=EasyEnsembleClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                        XGBClassifier(random_state=self.r) )
            
        self.name='XGB_EasyEnsemble_'+self.select+'_select'
        print(self.name)
        super().ML_model(log)
    def XGB_BalancedBagging(self,n):
        try:
            log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         XGBClassifier(random_state=self.r,tree_method='gpu_hist'))
            print(self.name)
            self.ML_model(log)
        except:
            log=BalancedBaggingClassifier(n_estimators=n,replacement=True,random_state=self.r,estimator = 
                                         XGBClassifier(random_state=self.r))
        self.name='XGB_BalancedBagging_'+self.select+'_select'
        print(self.name)
        super().ML_model(log)
 
'''
dataset=load_data('FIN+KLD')
model_log=Logestic(dataset,seed=0,year=2017,target='Brupt_crsp_t1',select='RF',n_years=1)
model_log.logistic_basic()  
model_log.logistic_BalancedBagging(10) 
model_log.logistic_EasyEnsemble(10)
'''    
