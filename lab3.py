import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from numpy import absolute,mean,std

df=pd.read_csv('NormalizedData.csv')

y=df['BikeBuyer'].ravel()
X=df.drop(columns=['BikeBuyer'])

X_train, X_test, y_train, y_test = train_test_split( X, y)


def SVM_Classification(TrainingData,TrainLabels,TestData,TestLabels,k='linear'):
    
    from sklearn.metrics import confusion_matrix
    Train=np.asarray(TrainingData)
    Labels=np.asarray(TrainLabels)
    
    Test=np.asarray(TestData)
    TestLabels=np.asarray(TestLabels)
    
    clf=SVC(kernel=k,gamma='scale')
    clf.fit(Train,Labels)
    
    prediction=clf.predict(Test)
    
    #true positve, true negative, false positive, false negative
    tn, fp, fn, tp = confusion_matrix(TestLabels, prediction).ravel()
    #accuracy
    acc=(tp+tn)/(tp+tn+fp+fn)
    acc=acc*100
    #false positive rate & miss rate
    fpr=fp/(fp+tn)
    missrate=fn/(tp+fn)
    
   
    
    
    
    
    print("---------------------------------------------------")
    print("For SVM classified with kernel={}".format(k))
    #print("Cross Validation of Train Dataset ")
    #scores = cross_val_score(clf, X, y, cv=5)
    # summarize the model performance
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    
    print("\nTest Data")
    print("\n Test Accuracy:{}".format(acc))
    print("False positive rate:{}".format(fpr))
    print("Miss Rate:{}".format(missrate))
    
    
    
    
    return acc,fpr,missrate



#trainig SVM with linear kernel
acc,fpr,missrate=SVM_Classification( X_train, y_train, X_test, y_test)

#training the SVM with kernel=rbf
acc,fpr,missrate=SVM_Classification(X_train, y_train, X_test, y_test,k='rbf')

#training the SVM with kernel=polynomial
acc,fpr,missrate=SVM_Classification(X_train, y_train, X_test, y_test,k='poly')


def NN_Classification(TrainingData,TrainLabels,TestData,TestLabels,hiddenLayerTupple=(12,10,5,2)):
    
    from sklearn.metrics import confusion_matrix
    Train=np.asarray(TrainingData)
    Labels=np.asarray(TrainLabels)
    
    Test=np.asarray(TestData)
    TestLabels=np.asarray(TestLabels)
    
    clf=MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=hiddenLayerTupple, random_state=1)
    clf.fit(Train,Labels)
    
    prediction=clf.predict(Test)
    
    #true positve, true negative, false positive, false negative
    tn, fp, fn, tp = confusion_matrix(TestLabels, prediction).ravel()
    #accuracy
    acc=(tp+tn)/(tp+tn+fp+fn)
    acc=acc*100
    #false positive rate & miss rate
    fpr=fp/(fp+tn)
    missrate=fn/(tp+fn)
    
    print("---------------------------------------------------")
    print("For NN classified with  hidden layers={} having {} neurons".format(len(hiddenLayerTupple),hiddenLayerTupple))
    
    #print("Cross Validation of Train Dataset ")
    #scores = cross_val_score(clf, X, y, cv=5)
    # summarize the model performance
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("\n Test Accuracy:{}".format(acc))
    print("False positive rate:{}".format(fpr))
    print("Miss Rate:{}".format(missrate))
    
    
    return acc,fpr,missrate



#trainig NN with 4 layers 
acc,fpr,missrate=NN_Classification( X_train, y_train, X_test, y_test,hiddenLayerTupple=(12,10,5))

#training the NN with 4 layers 
acc,fpr,missrate=NN_Classification(X_train, y_train, X_test, y_test,hiddenLayerTupple=(50,25,12))

#training the NN with 4 layers 
acc,fpr,missrate=NN_Classification(X_train, y_train, X_test, y_test,hiddenLayerTupple=(75,25,12,5))

def PCA_Analysis(data):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import plotly.express as px
    from plotly.offline import plot
    
    features=list(data.columns)
    data_array=np.asarray(data)
        
    print("Visualising 2D- PCA components")
    pca = PCA(n_components=2) # estimate only 2 PCs
    X_new = pca.fit_transform(data_array) # project the original data into the PCA space
    
    
    fig, axes = plt.subplots(1,2)
    axes[0].scatter(data_array[:,0], data_array[:,1], c=y)
    axes[0].set_xlabel('x1')
    axes[0].set_ylabel('x2')
    axes[0].set_title('Before PCA')
    axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].set_title('After PCA')
    plt.show()
    
    print("Visualising 5  PCA components")
   
    pca = PCA(n_components=5)
    components = pca.fit_transform(data[features])
    total_var = pca.explained_variance_ratio_.sum() * 100
    labels = {str(i): f"PC {i+1}" for i in range(5)}

    fig=px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(5),
        title=f'Total Explained Variance: {total_var:.2f}%',
        
    )
    fig.update_traces(diagonal_visible=False)
    fig.show()
    plot(fig)
    
    

    
    
    
PCA_Analysis(X)   