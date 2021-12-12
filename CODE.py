from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile,f_classif
from modAL import uncertainty
import math
from sklearn.metrics import accuracy_score
from sklearn import tree,svm
from sklearn.ensemble import RandomForestClassifier
import copy
from math import log
from sklearn.cluster import KMeans

def extract_from_mail(filename):
    finalList=[]
    fields=[]
    with open(filename,'r',encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        fields = next(csvreader)
        for row in csvreader:
            finalList.append(row)
    emails=[]
    labels=[]
    for row in finalList:
        emails.append(row[0])
        labels.append(row[1])
    return emails,labels

def  init():
    #POPULATING EMAILS AND CORRESPONDING LABELS
    filename="2017B5A70828P.csv"
    emails,labels=extract_from_mail(filename)
    dict={}
    dict['placements']=0
    dict['social']=1
    dict['promotions']=2
    dict['news']=3
    dict['acads']=4
    dict['misc']=5
    new_labels=[]
    for x in labels:
        new_labels.append(dict[x])
    #splitting the data set into training and test data
    emails_train,emails_test,labels_train,labels_test = train_test_split(emails,new_labels,test_size=0.3,random_state=42)

    #transforming emails into word count
    vectorizer=TfidfVectorizer(stop_words='english')
    emails_train_transformed=vectorizer.fit_transform(emails_train)
    emails_test_transformed=vectorizer.transform(emails_test)

    #selecting top 20% features to reduce dimension
    selector = SelectPercentile(f_classif, percentile=20)
    selector.fit(emails_train_transformed,labels_train)
    emails_train_transformed=selector.transform(emails_train_transformed).toarray()
    emails_test_transformed=selector.transform(emails_test_transformed).toarray()

    return emails_train_transformed,labels_train,emails_test_transformed,labels_test

def activeLearnerUncertainty(percentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active):
    n=math.ceil((len(labels_train_0)+len(labels_train_active))*percentInFraction)

    #creating the initial classifier
    #emails_train_0,emails_train_active,labels_train_0,labels_train_active=train_test_split(emails_train,labels_train,train_size=0.1,random_state=42)
    ret=[]
    # print(len(emails_train_0))
    # print(len(labels_train_0))
    # print()
    # classifier.fit(emails_train_0,labels_train_0)
    # pred=classifier.predict(emails_test)
    #ret.append(accuracy_score(pred,labels_test))

    emails_train_1=emails_train_0
    labels_train_1=copy.deepcopy(labels_train_0)
    arr=uncertainty.uncertainty_sampling(classifier=classifier,X=emails_train_active,n_instances=n)
    for x in arr:
        emails_train_1=np.insert(emails_train_1,len(emails_train_1),emails_train_active[x],axis=0)
        labels_train_1.append(labels_train_active[x])
    classifier.fit(emails_train_1,labels_train_1)
    pred=classifier.predict(emails_test)
    ret.append(accuracy_score(pred,labels_test))

    emails_train_2=emails_train_0
    labels_train_2=copy.deepcopy(labels_train_0)
    arr=uncertainty.margin_sampling(classifier=classifier,X=emails_train_active,n_instances=len(emails_train_active))
    #smallest margin sampling
    for i in range(n):
        emails_train_2=np.insert(emails_train_2,len(emails_train_2),emails_train_active[arr[i]],axis=0)
        labels_train_2.append(labels_train_active[arr[i]])
    classifier.fit(emails_train_2,labels_train_2)
    pred=classifier.predict(emails_test)
    ret.append(accuracy_score(pred,labels_test))
    #largest margin sampling
    emails_train_4=emails_train_0
    labels_train_4=copy.deepcopy(labels_train_0)
    arr=uncertainty.margin_sampling(classifier=classifier,X=emails_train_active,n_instances=len(emails_train_active))
    for i in range(len(labels_train_active)-n,len(labels_train_active)):
        emails_train_4=np.insert(emails_train_4,len(emails_train_4),emails_train_active[arr[i]],axis=0)
        labels_train_4.append(labels_train_active[arr[i]])
    classifier.fit(emails_train_4,labels_train_4)
    pred=classifier.predict(emails_test)
    ret.append(accuracy_score(pred,labels_test))

    emails_train_3=emails_train_0
    labels_train_3=copy.deepcopy(labels_train_0)
    arr=uncertainty.entropy_sampling(classifier=classifier,X=emails_train_active,n_instances=n)
    for x in arr:
        emails_train_3=np.insert(emails_train_3,len(emails_train_3),emails_train_active[x],axis=0)
        labels_train_3.append(labels_train_active[x])
    classifier.fit(emails_train_3,labels_train_3)
    pred=classifier.predict(emails_test)
    ret.append(accuracy_score(pred,labels_test))
    return ret

def CommitteeKLasc(learners,n_instances,X_active,Y_active,probs):
    labels=[]
    for learner in learners:
        labels.append(learner.predict(X_active))
    prob=[]
    for i in range(len(X_active)):
        temp=[0,0,0,0,0,0]
        for j in range(len(learners)):
            temp[labels[j][i]]+=1
        for j in range(len(temp)):
            temp[j]/=len(learners)
        prob.append(temp)
    ret=[]
    for i in prob:
        temp=0
        for j in range(len(i)):
            if i[j]!=0:
                temp+=i[j]*log(probs[j]*i[j])
        ret.append(temp)
    
    sorted_X=X_active[np.array(ret).argsort()]
    sorted_Y=[x for _, x in sorted(zip(ret,Y_active))]
    # sorted_X=np.flipud(sorted_X)
    # sorted_Y.reverse()
    return sorted_X[:n_instances-1],sorted_Y[:n_instances-1],prob

    
def activeLearnerQBC(percentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active):    
    n=math.ceil((len(labels_train_0)+len(labels_train_active))*percentInFraction)
    ret=[]

    c1=tree.DecisionTreeClassifier().fit(emails_train_0,labels_train_0)
    c2=RandomForestClassifier(n_estimators=10,max_depth=1000).fit(emails_train_0,labels_train_0)
    c3=svm.SVC(kernel='linear', C=1,decision_function_shape='ovo').fit(emails_train_0,labels_train_0)
    c4=svm.SVC(kernel='rbf', gamma=1, C=1,decision_function_shape='ovo').fit(emails_train_0,labels_train_0)
    c5=svm.SVC(kernel='sigmoid', C=1,decision_function_shape='ovo').fit(emails_train_0,labels_train_0)

    #vote entropy sampling
    probs=[1]*6
    emails,labels,temp=CommitteeKLasc([c1,c2,c3,c4,c5],5,emails_train_active,labels_train_active,probs)
    emails_train_1=np.append(emails_train_0,emails,axis=0)
    labels_train_1=copy.deepcopy(labels_train_0)
    labels_train_1.extend(labels)
    classifier.fit(emails_train_1,labels_train_1)
    pred=classifier.predict(emails_test)
    ret.append(accuracy_score(pred,labels_test))

    #KL-Divergence algorithm
    probs=[6]*6
    emails,labels,temp=CommitteeKLasc([c1,c2,c3,c4,c5],5,emails_train_active,labels_train_active,probs)
    emails_train_1=np.append(emails_train_0,emails,axis=0)
    labels_train_1=copy.deepcopy(labels_train_0)
    labels_train_1.extend(labels)
    classifier.fit(emails_train_1,labels_train_1)
    pred=classifier.predict(emails_test)
    ret.append(accuracy_score(pred,labels_test))

    return ret

def passiveLearner(percentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active):
    emails_train_passive=emails_train_0
    labels_train_passive=copy.deepcopy(labels_train_0)
    if(percentInFraction>0):
        emails_train,t1,labels_train,t2=train_test_split(emails_train_active,labels_train_active,train_size=percentInFraction)
        for x in range(len(emails_train)):
            emails_train_passive=np.insert(emails_train_passive,len(emails_train_passive),emails_train[x],axis=0)
            labels_train_passive.append(labels_train[x])
    classifier.fit(emails_train_passive,labels_train_passive)
    pred=classifier.predict(emails_test)
    return accuracy_score(pred,labels_test)

def K_MeanClusteredLearning(emails_train_active,labels_train_active):
    emails_Tocluster,t1,labels_Tocluster,t2=train_test_split(emails_train_active,labels_train_active,train_size=0.4)
    clusterAlgo=KMeans(n_clusters=6)
    clusters=clusterAlgo.fit_predict(emails_Tocluster)
    clusterLabels=[]
    for i in range(6):
        clusterLabels.append([])
    for i in range(len(clusters)):
        clusterLabels[clusters[i]].append(labels_Tocluster[i])
    print(clusterLabels)
    #select 20% from each cluster and label:
    labelsForEachCluster=[]
    for arr in clusterLabels:
        selected,t1=train_test_split(arr,train_size=0.2)
        temp=[0,0,0,0,0,0]
        count=0
        label=-1
        for i in selected:
            temp[i]+=1
            if(temp[i]>count):
                count=temp[i]
                label=i
        labelsForEachCluster.append(label)
    pred=[]
    for obj in clusters:
        pred.append(labelsForEachCluster[obj])
    return accuracy_score(pred,labels_Tocluster)

def mainUS(classifier,emails_train_0,emails_train_active,labels_train_0,labels_train_active):
    x=[0.1,0.2,0.3,0.4,0.5]
    pts=[]
    for i in range(5):
        pts.append([])
    val=passiveLearner(0,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
    for i in range(5):
        pts[i].append(val)
    for precentInFraction in x:
        arr=activeLearnerUncertainty(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
        for i in range(4):
            pts[i].append(arr[i])
        pts[4].append(passiveLearner(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active))
    xLabel=[0.0]
    xLabel.extend(x)
    fig,ax=plt.subplots()
    for y in pts:
        ax.plot(xLabel,y,marker='o')
    ax.set_xlabel("Percentage of Data Taken for Active Learning")
    ax.set_ylabel("Accuracy")
    ax.legend(['Least Confidence','Smallest Margin Sampling','Largest Margin Sampling','Entropy Sampling','Passive Learning'])
    plt.title("Percentage of Data Taken for Active Learning v/s Accuracy for Polynomial Kernel SVM Classifier")
    plt.show()
       
def mainQBC(classifier,emails_train_0,emails_train_active,labels_train_0,labels_train_active):
    x=[0.1,0.2,0.3,0.4,0.5]
    pts=[]
    for i in range(3):
        pts.append([])
    val=passiveLearner(0,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
    for i in range(3):
        pts[i].append(val)
    # arr=activeLearnerQBC(0.5,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
    # print(arr)
    for precentInFraction in x:
        arr=activeLearnerQBC(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
        for i in range(2):
            pts[i].append(arr[i])
        pts[2].append(passiveLearner(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active))
    print(pts)
    xLabel=[0.0]
    xLabel.extend(x)
    fig,ax=plt.subplots()
    for y in pts:
        ax.plot(xLabel,y,marker='o')
    ax.set_xlabel("Percentage of Data Taken for Active Learning")
    ax.set_ylabel("Accuracy")
    ax.legend(['Vote Entropy','KL Divergence','Passive Learning'])
    plt.title("Percentage of Data Taken for Active Learning v/s Accuracy for Polymnomial Kernel SVM Classifier")
    plt.show()

#This is the Version Space part. The below 2 functions. CommitteeKLasc method returns the probabily array of all tuples. The number of classes are 6.
def greedyApproach(percentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active):
    n=math.ceil((len(labels_train_0)+len(labels_train_active))*percentInFraction)

    c1=tree.DecisionTreeClassifier().fit(emails_train_0,labels_train_0)
    c2=RandomForestClassifier(n_estimators=10,max_depth=1000).fit(emails_train_0,labels_train_0)
    c3=svm.SVC(kernel='linear', C=1,decision_function_shape='ovo').fit(emails_train_0,labels_train_0)
    c4=svm.SVC(kernel='rbf', gamma=1, C=1,decision_function_shape='ovo').fit(emails_train_0,labels_train_0)
    c5=svm.SVC(kernel='sigmoid', C=1,decision_function_shape='ovo').fit(emails_train_0,labels_train_0)

    probs=[1]*6
    grey=[]
    for i in range(len(labels_train_active)):
        grey.append(False)
    emails,labels,prob=CommitteeKLasc([c1,c2,c3,c4,c5],5,emails_train_active,labels_train_active,probs)

    i=0
    emails_train_1=emails_train_0
    labels_train_1=copy.deepcopy(labels_train_0)
    while n>0 and i<len(prob):
        flag=False
        temp=prob[i]
        for x in temp:
            if x>0 and x<1:
                flag=True
        if flag:
            emails_train_1=np.insert(emails_train_1,len(emails_train_1),emails_train_active[i],axis=0)
            labels_train_1.append(labels_train_active[i])
            n-=1
            grey[i]=True
        i+=1

    i=0
    while n>0:
        if not grey[i]:
            emails_train_1=np.insert(emails_train_1,len(emails_train_1),emails_train_active[i],axis=0)
            labels_train_1.append(labels_train_active[i])
            n-=1
        i+=1 

    classifier.fit(emails_train_1,labels_train_1)
    return(accuracy_score(labels_test,classifier.predict(emails_test)))

def mainGreedyApproach(classifier,emails_train_0,emails_train_active,labels_train_0,labels_train_active):
    x=[0.1,0.2,0.3,0.4,0.5]
    pts=[]
    for i in range(2):
        pts.append([])
    val=passiveLearner(0,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
    for i in range(2):
        pts[i].append(val)

    for precentInFraction in x:
        arr=greedyApproach(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
        pts[0].append(arr)
        pts[1].append(passiveLearner(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active))
    print(pts)
    xLabel=[0.0]
    xLabel.extend(x)
    fig,ax=plt.subplots()
    for y in pts:
        ax.plot(xLabel,y,marker='o')
    ax.set_xlabel("Percentage of Data Taken for Active Learning")
    ax.set_ylabel("Accuracy")
    ax.legend(['Greedy Approach','Passive Learning'])
    plt.title("Percentage of Data Taken for Active Learning v/s Accuracy for Polynomial Kernel SVM Classifier")
    plt.show()

def mainCombined(classifier,emails_train_0,emails_train_active,labels_train_0,labels_train_active):
    x=[0.1,0.2,0.3,0.4,0.5]
    pts=[]
    for i in range(3):
        pts.append([])
    val=passiveLearner(0,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
    for i in range(3):
        pts[i].append(val)

    for precentInFraction in x:
        arr=activeLearnerUncertainty(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
        pts[0].append(max(arr))
        arr=activeLearnerQBC(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active)
        pts[1].append(max(arr))
        pts[2].append(passiveLearner(precentInFraction,classifier,emails_train_0,labels_train_0,emails_test,labels_test,emails_train_active,labels_train_active))
    print(pts)
    xLabel=[0.0]
    xLabel.extend(x)
    fig,ax=plt.subplots()
    for y in pts:
        ax.plot(xLabel,y,marker='o')
    ax.set_xlabel("Percentage of Data Taken for Active Learning")
    ax.set_ylabel("Accuracy")
    ax.legend(['Uncertainty Sampling','Query by Committee','Passive Learning'])
    plt.title("Percentage of Data Taken for Active Learning v/s Accuracy for Polynomial Kernel SVM Classifier")
    plt.show()

def mainKMCL():
    emails_train,labels_train,emails_test,labels_test=init()
    emails_train_0,emails_train_active,labels_train_0,labels_train_active=train_test_split(emails_train,labels_train,train_size=0.1) 
    print(K_MeanClusteredLearning(emails_train_active,labels_train_active))

emails_train,labels_train,emails_test,labels_test=init()
emails_train_0,emails_train_active,labels_train_0,labels_train_active=train_test_split(emails_train,labels_train,train_size=0.1)
mainKMCL()
#classifier=svm.SVC(kernel='poly', C=1,decision_function_shape='ovo',probability=True)
#mainQBC(classifier,emails_train_0,emails_train_active,labels_train_0,labels_train_active)