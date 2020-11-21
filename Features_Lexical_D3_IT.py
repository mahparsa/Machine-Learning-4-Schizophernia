#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:51:21 2019

@author: mahparsa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 13:57:17 2019

@author: mahparsa

This aim to calculate the similairty between sentenses of an interview.
It has all features
"""
import nltk
import numpy as np
from nltk.corpus import PlaintextCorpusReader
from nltk.corpus import gutenberg
import pandas as pd
import nltk
import gensim
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
import nltk
nltk.download('wordnet')
import nltk
from gensim import utils 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize 
from gensim.parsing.preprocessing import stem_text
from nltk.stem import PorterStemmer
import collections
from collections import Counter
from stemming.porter2 import stem
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from numpy import array
nltk.download('averaged_perceptron_tagger')
stop_words = set(stopwords.words('english')) 
#from nltk.stem import LancasterStemmer
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from difflib import SequenceMatcher

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def percentage(count, total): 
     return 100 * count / total  

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

import matplotlib.pyplot as plt    
from nltk.corpus import stopwords  

def lexical_diversity(text): 
    return len(set(text)) / len(text)   

def READ_INT( parameters ):
   "use the root and read files and make a list of that"
   corpus_root = parameters # Mac users should leave out C:
   corpus = PlaintextCorpusReader(corpus_root, '.*txt')  #
   doc = pd.DataFrame(columns=['string_values'])
   for filename in corpus.fileids(): 
       value1=corpus.raw(filename)
       doc = doc.append({'string_values': value1}, ignore_index=True)
   docs=doc.values.tolist()
   return [docs]
def Pre_Word( doc ):
    #provide a list of words.
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    m=str(doc)
    mm=m.lower()
    mmm=lemmatizer.lemmatize(mm)
    return [mmm]

from nltk import sent_tokenize
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def get_cosine_sim(*strs): 
    vectors = [t for t in get_vectors(*strs)]
    return cosine_similarity(vectors) 

#docs_A=READ_INT('SZ_QA_3')

docs_A=READ_INT('SZ_A_7')
def Coherence_M_4(docs_A): 
    #measure the lexical richness
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Co_2=lexical_diversity(doc)
        Coh_M=np.append(  Coh_M,  Co_2)
    return[Coh_M] 

#    
##Brunet Index
#
def Coherence_M_10(docs_A): 
    stop_words = set(stopwords.words('english')) 
    #Measure the Brunet Index
    #after removing stop words 
    stemmer = nltk.PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    Lexical_BI=np.array([])
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        doc=doc.lower()
        doc=lemmatizer.lemmatize(doc)
        Sent_doc=sent_tokenize(doc)
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        #word_tokens = [w for w in tokenized_word if w.isalpha() and not in stop_words ] 
        word_tokens = [w for w in tokenized_word if w.isalpha() and not w in stop_words  ] 
        
        BI_N = [stemmer.stem(w) for w in  word_tokens]
        #BI_N = word_tokens
        #BI_U = sorted (set([stemmer.stem(verb) for verb in  word_tokens]))
        BI_U = sorted (set(BI_N))
        BI_F=len(BI_N) ** (len(BI_U) ** -0.165)
        Lexical_BI=np.append(Lexical_BI, BI_F)
        Coh_M=Lexical_BI/sum(Lexical_BI)
    return[Coh_M]             
#    
#Honore Satistic
        
def Coherence_M_11(docs_A): 
    #Measure Honore Satistic
    #after removing stop words 
    import math 
    stemmer = nltk.PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    Lexical_HS=[]
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        doc=doc.lower()
        doc=lemmatizer.lemmatize(doc)
        Sent_doc=sent_tokenize(doc)
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha() and not w in stop_words  ] 
        N = [stemmer.stem(verb) for verb in word_tokens]
        fdist = FreqDist(N)
        Words_freq =np.asarray([fdist[w] for w in  N ])
        One_Time_Words=np.asarray(np.where(Words_freq==1))
        h=np.array(N)
        N_1 =len(h[One_Time_Words])
        U = len(sorted (set([stemmer.stem(verb) for verb in  word_tokens])))
        Honore_Satistic_Index=(100*math.log10(len(N)))/(1-((N_1)/U))
        #math.log10(N)
        Lexical_HS=np.append(Lexical_HS, Honore_Satistic_Index)
        Coh_M=Lexical_HS/sum(Lexical_HS)
    return[Coh_M]                 

##------   
##Readability of Transcripts 
# 
##1.Flesch Reading Score
#
def Coherence_M_12(docs_A): 
    #Measure Readability 
    #after removing stop words 
    import syllables
    Factor_1=206.835
    Factor_2=1.015
    Factor_3=84.6
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        doc=str(MyDoc[k])
        doc=doc.lower()
        Sent_doc=sent_tokenize(doc)
        T_Sent=len( Sent_doc) #Total Sentenses
        T_Word=np.array([])
        T_Syll=np.array([])
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha() ] 
        Syllables_Word= [ syllables.estimate(w) for w in word_tokens ] 
        T_Syll= np.append(T_Syll, sum(Syllables_Word)) #Total Syllables
        T_Word=np.append(T_Word, len(word_tokens)) #Total words            
        M = Factor_1 - Factor_2* (sum(T_Word)/T_Sent) - Factor_3 * (sum(T_Syll)/sum(T_Word)) 
        Coh_M=np.append(  Coh_M,  M)
    return[Coh_M]                    
#2.Flesch-Kincaid Grade Level       

def Coherence_M_13(docs_A): 
    #Measure Flesch-Kincaid Grade Level  
    #after removing stop words 
    import syllables
    Factor_1=0.39
    Factor_2=11.8
    Factor_3=15.59
    Coh_M = np.array([])
    MyDoc = docs_A  
    Sim_Sent_Doc = []
    for k in range(len(MyDoc)):
        doc=[]        
        doc=str(MyDoc[k])
        Sent_doc=sent_tokenize(doc)
        T_Sent=len( Sent_doc) #Total Sentenses
        T_Word=np.array([])
        T_Syll=np.array([])
        tokenized_word=[]
        word_tokens=[]
        tokenized_word=word_tokenize(doc)
        word_tokens = [w for w in tokenized_word if w.isalpha() ] 
        Syllables_Word= [ syllables.estimate(w) for w in word_tokens ] 
        T_Syll= np.append(T_Syll, sum(Syllables_Word)) #Total Syllables
        T_Word=np.append(T_Word, len(word_tokens)) #Total words            
        M = Factor_1 * (sum(T_Word)/T_Sent) + Factor_2 * (sum(T_Syll)/sum(T_Word)) + Factor_3
        Coh_M=np.append(  Coh_M,  M)
    return[Coh_M]           
# 


#
No_Measure=5
No_Subject=15

h=np.zeros((No_Measure, No_Subject))
#for i in range(No_Measure):
    
h[0][:]=np.asarray(Coherence_M_4( docs_A[0]))
h[1][:]=np.asarray(Coherence_M_10( docs_A[0]))
h[2][:]=np.asarray(Coherence_M_11( docs_A[0]))
h[3][:]=np.asarray(Coherence_M_12( docs_A[0]))
h[4][:]=np.asarray(Coherence_M_13( docs_A[0]))
#h[5][:]=np.asarray(Coherence_M_6( docs_A[0]))
#h[6][:]=np.asarray(Coherence_M_7( docs_A[0]))
#h[7][:]=np.asarray(Coherence_M_8( docs_A[0]))
#h[8][:]=np.asarray(Coherence_M_9( docs_A[0]))
#h[9][:]=np.asarray(Coherence_M_10( docs_A[0]))
#h[10][:]=np.asarray(Coherence_M_11( docs_A[0]))
#h[11][:]=np.asarray(Coherence_M_12( docs_A[0]))
#h[12][:]=np.asarray(Coherence_M_13( docs_A[0]))
#h[13][:]=np.asarray(Coherence_M_14( docs_A[0]))
#h[14][:]=np.asarray(Coherence_M_15( docs_A[0]))
##h[15][:]=np.asarray(Coherence_M_16( docs_A[0]))

#https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html



plt.plot(h)
plt.plot(np.transpose(h))
hh=np.transpose(h)
X = np.transpose(h)
Y = np.asarray([0, 0 ,0,0, 0, 1 ,1,1, 1, 1,1,1])

plt.plot(h)
plt.plot(np.transpose(h))
hh=np.transpose(h)

dataset = pd.DataFrame({'Lexical_Diversity':hh[:,0],'Brunet_Index':hh[:,1],'Honore_Satistic':hh[:,2],'Flesch Reading':hh[:,3], 'Flesch-Kincaid':hh[:,4] })
                
SZ_Type = ['Incoherence', 'Incoherence','Incoherence','Incoherence','Incoherence', 'Tangentiality', 'Tangentiality', 'Tangentiality', 'Tangentiality', 'Tangentiality', 'Tangentiality', 'Tangentiality', 'Tangentiality', 'Tangentiality','Tangentiality' ]                      
dataset['SZ_Type']=SZ_Type
                  

import seaborn as sns
sns.set(style="ticks")
#
sns.pairplot(dataset, hue='SZ_Type', size=1.75)

from yellowbrick.features import Rank2D
features = ['Lexical_Diversity','Brunet_Index', 'Honore_Satistic', 'Flesch Reading', 'Flesch-Kincaid']                         

# Instantiate the visualizer with the Pearson ranking algorithm
visualizer = Rank2D(features=features, algorithm='pearson')
X=np.transpose(h)
Y = np.asarray([0, 0 ,0, 0, 0, 1,1,1, 1,1,1, 1,1,1,1])


visualizer.fit(X, Y)                # Fit the data to the visualizer
visualizer.transform(X)             # Transform the data
visualizer.poof()           



from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer


from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors = 5)
clf=knn
scores_A = cross_val_score(clf, X, Y, cv=3)
scores_F = cross_val_score(clf, X, Y, cv=3,scoring='f1_macro')
scores_P = cross_val_score(clf, X, Y, cv=3,scoring='precision_macro')
scores_R = cross_val_score(clf, X, Y, cv=3,scoring='recall_macro')
print("KNeighborsClassifier")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_A.mean(), scores_A.std() * 2))
print("f1_macro: %0.2f (+/- %0.2f)" % (scores_F.mean(), scores_F.std() * 2))
print("precision_macro: %0.2f (+/- %0.2f)" % (scores_P.mean(), scores_P.std() * 2))
print("recall_macro: %0.2f (+/- %0.2f)" % (scores_R.mean(), scores_R.std() * 2))

from sklearn.model_selection import cross_val_score    
from sklearn.linear_model import LogisticRegression
#create a new logistic regression model
log_reg = LogisticRegression()
clf=[]
clf=log_reg
scores_A = cross_val_score(clf, X, Y, cv=3)
scores_F = cross_val_score(clf, X, Y, cv=3,scoring='f1_macro')
scores_P = cross_val_score(clf, X, Y, cv=3,scoring='precision_macro')
scores_R = cross_val_score(clf, X, Y, cv=3,scoring='recall_macro')
print("LogisticRegression")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_A.mean(), scores_A.std() * 2))
print("f1_macro: %0.2f (+/- %0.2f)" % (scores_F.mean(), scores_F.std() * 2))
print("precision_macro: %0.2f (+/- %0.2f)" % (scores_P.mean(), scores_P.std() * 2))
print("recall_macro: %0.2f (+/- %0.2f)" % (scores_R.mean(), scores_R.std() * 2))

from sklearn.svm import SVC 
svm = SVC(kernel = 'rbf', C = 0.5)
clf=[]
clf=svm
scores_A = cross_val_score(clf, X, Y, cv=3)
scores_F = cross_val_score(clf, X, Y, cv=3,scoring='f1_macro')
scores_P = cross_val_score(clf, X, Y, cv=3,scoring='precision_macro')
scores_R = cross_val_score(clf, X, Y, cv=3,scoring='recall_macro')
print("svm")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_A.mean(), scores_A.std() * 2))
print("f1_macro: %0.2f (+/- %0.2f)" % (scores_F.mean(), scores_F.std() * 2))
print("precision_macro: %0.2f (+/- %0.2f)" % (scores_P.mean(), scores_P.std() * 2))
print("recall_macro: %0.2f (+/- %0.2f)" % (scores_R.mean(), scores_R.std() * 2))


from sklearn.svm import SVC 
svm = SVC(kernel = 'linear')
clf=[]
clf=svm
scores_A = cross_val_score(clf, X, Y, cv=3)
scores_F = cross_val_score(clf, X, Y, cv=3,scoring='f1_macro')
scores_P = cross_val_score(clf, X, Y, cv=3,scoring='precision_macro')
scores_R = cross_val_score(clf, X, Y, cv=3,scoring='recall_macro')
print("svm-linear")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_A.mean(), scores_A.std() * 2))
print("f1_macro: %0.2f (+/- %0.2f)" % (scores_F.mean(), scores_F.std() * 2))
print("precision_macro: %0.2f (+/- %0.2f)" % (scores_P.mean(), scores_P.std() * 2))
print("recall_macro: %0.2f (+/- %0.2f)" % (scores_R.mean(), scores_R.std() * 2))

from sklearn.ensemble import ExtraTreesClassifier 
clf=[]
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0) 
scores_A = cross_val_score(clf, X, Y, cv=3)
scores_F = cross_val_score(clf, X, Y, cv=3,scoring='f1_macro')
scores_P = cross_val_score(clf, X, Y, cv=3,scoring='precision_macro')
scores_R = cross_val_score(clf, X, Y, cv=3,scoring='recall_macro')
print("ExtraTreesClassifier")

print("Accuracy: %0.2f (+/- %0.2f)" % (scores_A.mean(), scores_A.std() * 2))
print("f1_macro: %0.2f (+/- %0.2f)" % (scores_F.mean(), scores_F.std() * 2))
print("precision_macro: %0.2f (+/- %0.2f)" % (scores_P.mean(), scores_P.std() * 2))
print("recall_macro: %0.2f (+/- %0.2f)" % (scores_R.mean(), scores_R.std() * 2))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)
clf=[]
clf = RF
scores_A = cross_val_score(clf, X, Y, cv=3)
scores_F = cross_val_score(clf, X, Y, cv=3,scoring='f1_macro')
scores_P = cross_val_score(clf, X, Y, cv=3,scoring='precision_macro')
scores_R = cross_val_score(clf, X, Y, cv=3,scoring='recall_macro')
print("RandomForestClassifier")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_A.mean(), scores_A.std() * 2))
print("f1_macro: %0.2f (+/- %0.2f)" % (scores_F.mean(), scores_F.std() * 2))
print("precision_macro: %0.2f (+/- %0.2f)" % (scores_P.mean(), scores_P.std() * 2))
print("recall_macro: %0.2f (+/- %0.2f)" % (scores_R.mean(), scores_R.std() * 2))