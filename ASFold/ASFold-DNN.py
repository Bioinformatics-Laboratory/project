import numpy as np
import pandas as pd
import math
from xgboost import XGBClassifier #  xgboost
import lightgbm as lgb
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
# 引用 scikit-learn 评估方法
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef, confusion_matrix, classification_report
# 引用模型评估方法
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel,SelectKBest, f_classif,chi2
#SelectFromModel包括：L1-based feature selection 、 Tree-based feature selection 
#coef_（系数）适用于线性模型，而无系数的非线性模型使用feature_importances_
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,LassoCV,Ridge,ElasticNet,ElasticNetCV
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler,scale
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json, model_from_yaml
from keras.layers import Dense,Activation,Dropout
import keras.layers.advanced_activations
from keras import regularizers
from keras.layers.normalization import BatchNormalization
#regularizers.l1(0.001)#L1正则化
#regularizers.l2(0.001)#L2正则化
#regularizers.l1_l2(l1=0.001,l2=0.001)#L1和L2同时
from keras.wrappers.scikit_learn import KerasClassifier
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import LeaveOneOut
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, partial, Trials
from hyperopt.mongoexp import MongoTrials
from hyperas import optim
from hyperas.distributions import choice, uniform
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)

###############################################################分训练集和测试集###############################################################
def read_xy(PATH):
	dataset=pd.read_csv(PATH)
	col=dataset.columns.values.tolist()
	col1=col[1:]
	X=np.array(dataset[col1])
	y=dataset['class']
	split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
	for train_index, test_index in split.split(X, y):
		X_train, X_test=X[train_index], X[test_index]
		y_train, y_test=y[train_index], y[test_index]
		pd.DataFrame(X_train, columns=col1).to_csv("/ASTRAL186/train1/SXG/SXG_5_train.csv")
		pd.DataFrame(X_test, columns=col1).to_csv("/ASTRAL186/test1/SXG/SXG_5_test.csv")	
		y_train.to_csv("/ASTRAL186/train1/SXG/SXG_5_ytrain.csv", index=False)
		y_test.to_csv("/ASTRAL186/test1/SXG/SXG_5_ytest.csv", index=False)
	return y_test

if __name__ == '__main__':
	PATH='/ASTRAL186/train1/SXG/SXG_hmm_5_filter.csv'
	y_test=read_xy(PATH)
	print(y_test)

###############################################################ASFOLD-DNN第一步###############################################################
def read_xy(datapath):
    dataset=pd.read_csv(datapath)
    col=dataset.columns.values.tolist()
    col1=col[1:]
    print(len(col1))
    X_train=np.array(dataset[col1])
    y_train=preprocessing.LabelEncoder().fit_transform(dataset['class'])
    print(len(y_train))
    scale=StandardScaler().fit(X_train)
    X_train=scale.transform(X_train)
    f_dim=X_train.shape[1]
    y_train=np_utils.to_categorical(y_train)
    return X_train, y_train, f_dim

def DNN():
	rs=KFold(n_splits=10, shuffle=True, random_state=0)
	cvscores=[]
	for train, test in rs.split(X, y):
		model=Sequential()
		#基本模型
		hidden_1=2*f_dim+1 #Kolmogorov
		model.add(Dense(hidden_1, input_dim=f_dim, activation='relu'))#l2(0.001)的意思是该层权重矩阵的每个系数都会使网络总损失增加0.001*weight_coefficient_value
		model.add(Dropout(0.5))

		model.add(Dense(186, activation='softmax'))
		optimizer=Adam()
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X[train], y[train], epochs=10, batch_size=50, verbose=0)
		scores=model.evaluate(X[test], y[test], verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	return np.mean(cvscores)

if __name__ == '__main__':
	PATH=input(":")
	X, y, f_dim=read_xy(PATH)
	acc=DNN()

###############################################################ASFOLD-DNN第二步###############################################################
def read_xy(PATH):
	dataset= pd.read_csv(PATH) #用pandas读取原始数据
	col = dataset.columns.values.tolist()#取第一行
	col1 =col[1:] #取特征
	print(len(col1)) #特征维数
	X_train= np.array(dataset[col1])#取数据
	y_train=preprocessing.LabelEncoder().fit_transform(dataset['class']) #标签标准化
	print(len(y_train))
	#标准化
	scale = StandardScaler().fit(X_train)#特征矩阵标准化（与距离计算无关的概率模型、与距离计算无关的基于树的模型不需要）
	X_train = scale.transform(X_train)
	
	#带L1/L2/L1+L2惩罚项的逻辑回归作为基模型的特征选择——SelectFromModel
	#小的C会导致少的特征被选择。使用Lasso，alpha的值越大，越少的特征会被选择。
	######################################针对clf.coef_：1*n_features#####################################
	'''
	#clf=Lasso(normalize=True,alpha=0.001,max_iter=5000,random_state=0)#Lasso回归
	#clf = LassoCV()
	#clf=Ridge(normalize=True,alpha=0.001,max_iter=5000,random_state=0)#岭回归
	#clf=ElasticNet(normalize=True,alpha=0.001,l1_ratio=0.1,max_iter=5000,random_state=0)#弹性网络正则
	clf=LinearRegression(normalize=True)
	clf.fit(X_train, y_train)
	#print(clf.coef_)
	importance=np.abs(clf.coef_)
	#print(importance)
	'''
	######################################针对clf.coef_：n_classes*n_features#####################################
	
	#‘newton-cg’，‘sag’和‘lbfgs’等solvers仅支持‘L2’regularization，
	#‘liblinear’ solver同时支持‘L1’、‘L2’regularization，
	#若dual=Ture，则仅支持L2 penalty。
	clf=LogisticRegression(penalty='l1',C=0.1,solver='liblinear',random_state=0)#clf.coef_：n_classes*n_features
	#clf=LogisticRegression(penalty='l2',C=0.1,random_state=0)
	#clf=LR(threshold=0.5, C=0.1)#参数threshold为权值系数之差的阈值
	#clf=LinearSVC(penalty='l1',C=0.1,dual=False,random_state=0)
	#clf=LinearSVC(penalty='l2',C=0.1,random_state=0)
	clf.fit(X_train, y_train)
	#print(clf.coef_)
	#每个类别--每个属性--都有一个权重，将不同类别同一属性权重相加--即为该维度的--重要程度得分
	#方法一：
	importance=np.linalg.norm(clf.coef_,axis=0,ord=1) 
	#方法二：
	#coef=np.abs(clf.coef_)
	#importance=np.sum(coef,axis=0)
	#print(importance)
	
	mean=np.mean(importance)
	#print(mean)
	#median=np.median(importance)
	#print(median)

	#model=SelectFromModel(clf,prefit=True)
	model=SelectFromModel(clf,prefit=True,threshold=2.0*mean)
	'''
	model=SelectFromModel(estimator=clf).fit(X_train, y_train)
	importance=model.estimator_.coef_
	threshold=model.threshold_
	print(threshold)
	'''
	#threshold ： 阈值，string, float, optional default None
	#可以使用：median 或者 mean 或者 1.25 * mean 这种格式。
	#如果使用参数惩罚设置为L1，则使用的阈值为1e-5，否则默认使用用mean
	X_train=model.transform(X_train)
	f_dim=X_train.shape[1]
	print(f_dim)
	y_train=np_utils.to_categorical(y_train)
	return X_train, y_train, f_dim

def DNN():
	rs=KFold(n_splits=10, shuffle=True, random_state=0)
	cvscores=[]
	for train, test in rs.split(X, y):
		model=Sequential()
		#基本模型
		hidden_1=2*f_dim+1 #Kolmogorov
		model.add(Dense(hidden_1, input_dim=f_dim, activation='relu'))#l2(0.001)的意思是该层权重矩阵的每个系数都会使网络总损失增加0.001*weight_coefficient_value
		model.add(Dropout(0.5))

		model.add(Dense(186, activation='softmax'))
		optimizer=Adam()
		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
		#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		model.fit(X[train], y[train], epochs=10, batch_size=50, verbose=0)
		scores=model.evaluate(X[test], y[test], verbose=0)
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	return np.mean(cvscores)

if __name__ == '__main__':
	PATH=input(":")
	X, y ,f_dim=read_xy(PATH)
	acc=DNN()

###############################################################ASFOLD-DNN第三步###############################################################
def data():
    PATH="/ASTRAL186/train1/astral_train.csv"
    dataset=pd.read_csv(PATH)
    col=dataset.columns.values.tolist()
    col1=col[1:]
    print(len(col1))#2460
    X_train=np.array(dataset[col1])
    y_train=preprocessing.LabelEncoder().fit_transform(dataset['class'])
    print(len(y_train))#5273
    scale=StandardScaler().fit(X_train)
    X_train=scale.transform(X_train)

    PATH_="/ASTRAL186/test1/astral_test.csv"
    dataset_=pd.read_csv(PATH_)
    col_= dataset_.columns.values.tolist()
    col1_=col_[1:]
    print(len(col1_))#2460
    X_test= np.array(dataset_[col1_])
    y_test=preprocessing.LabelEncoder().fit_transform(dataset_['class'])
    print(len(y_test))#1319
    scale=StandardScaler().fit(X_test)
    X_test=scale.transform(X_test)

	#fs
    #clf=LogisticRegression(penalty='l1',C=0.1,solver='liblinear',random_state=0)########################edd+1.25mean(676)/tg+1.25mean(584)/astral+1.25mean(549)/astral_train+1.0*mean(867)
    clf=LinearSVC(penalty='l1',C=0.1,dual=False,random_state=0)########################dd+1.5mean(584)/le+mean(554)/astral_train1+1.25mean(794)
    clf.fit(X_train, y_train)
    importance=np.linalg.norm(clf.coef_,axis=0,ord=1) 
    mean=np.mean(importance)
    model=SelectFromModel(clf,prefit=True,threshold=1.25*mean)##########################
    
    X_train1=model.transform(X_train)
    print(X_train1.shape[1])#867/794
    X_test1=model.transform(X_test)
    print(X_test1.shape[1])#867/794
    
    y_train=np_utils.to_categorical(y_train)
    y_test=np_utils.to_categorical(y_test)
    
    return X_train1, X_test1, y_train, y_test

def create_model(X_train1, X_test1, y_train, y_test):
    activation={{choice(['elu','selu','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear'])}}
    optimizer={{choice(['RMSprop','Adam','Adamax','Nadam'])}}
    init = {{choice(['uniform', 'lecun_uniform', 'glorot_uniform', 'glorot_normal','he_normal', 'he_uniform'])}}
    layers={{choice([2, 3])}}
    hidden_1={{uniform(600, 1000)}}
    hidden_2={{uniform(400, 800)}}
    hidden_3={{uniform(200, 600)}}
    dropout_1={{uniform(0.1, 0.6)}}
    epochs={{choice([10,20,30,40,50,60,70,80,90,100,110,120])}}
    batch_size={{choice([20,40,60,80,100,120])}}
    learning_rate={{uniform(0.001, 0.005)}}
    function_mappings={
        'RMSprop': RMSprop,
        'Adam': Adam,
        'Adamax': Adamax,
        'Nadam': Nadam
    }

    f_dim_1=X_train1.shape[1]
    rs=KFold(n_splits=10, shuffle=True, random_state=0)
    cvscores=[]
    for train, test in rs.split(X_train1, y_train):
        model=Sequential()
        model.add(InputLayer(input_shape=(f_dim_1))) 

        for i in range(int(layers)):
            name='layers_{0}'.format(i+1)
            hidden=eval('hidden_{0}'.format(i+1))
            model.add(Dense(int(hidden), activation=activation, kernel_initializer=init, name=name))
            model.add(Dropout(dropout_1)) 

        model.add(Dense(186, kernel_initializer=init, activation='softmax'))
        optimizer1=function_mappings[optimizer](lr=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer1, metrics=['accuracy'])
        model.fit(X_train1[train], y_train[train], epochs=epochs, batch_size=batch_size, verbose=0)
        scores_1=model.evaluate(X_train1[test], y_train[test], verbose=0)
        #print("%s: %.2f%%" % (model.metrics_names[1], scores_1[1]*100))
        cvscores.append(scores_1[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    accuracy=np.mean(cvscores)
    scores_2=model.evaluate(X_test1, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores_2[1]*100))
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}

trials=Trials()
best_run,best_model,space=optim.minimize(model=create_model,
                                        data=data,
                                        algo=tpe.suggest,
                                        max_evals=100,
                                        trials=trials,
                                        eval_space=True,
                                        return_space=True,
                                        #rseed=100
                                        )
X_train1, X_test1, y_train, y_test=data()
# root
from hyperas.utils import eval_hyperopt_space
# H5
H5_path = "/ASTRAL186/astral_train1_hyperas_1.h5"
param_path = "/ASTRAL186/astral_train1_hyperas_param_1.csv"
print("Best performing model chosen hyper-parameters:")
print(best_run)
best_model.save(H5_path)
#print(space)
param = []
for t, trial in enumerate(trials):
    vals = trial.get('misc').get('vals')
    print("Trial %s vals: %s" % (t, vals))
    print(eval_hyperopt_space(space, vals))
    param_csv = eval_hyperopt_space(space, vals)
    param.append(param_csv)
data_csv = pd.DataFrame(data=param)
data_csv.to_csv(param_path, index=0)

	