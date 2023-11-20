import sys
from pyspark import SparkConf, SparkContext
import json
import itertools
import time
from itertools import chain




def generateFreqitem(basket, ck, support_threshold):
    #Function to generate frequent itemsets from candidata itemsets
    #inputs are basket, candidate sets and support_threshold
    C1=[tuple(x) for x in ck]
    cnt={}  #dict to store count for each candidate
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1

    freq_item=[] #frequent item to extract items count>=support_threshold
    for key in cnt:
        if cnt[key] >= support_threshold:
            freq_item.append(key)
    return freq_item     #return frequent items from candidate set



def son2count_freq(basket,subsets):
    C1=[tuple(x) for x in subsets] #make sure to convert elemnet inside list to a tuple before starting to create dict count
    cnt={}
    for i in basket:
        for c in C1:
            if (set(c).issubset(i)):
                if c in cnt:
                    cnt[c] += 1
                else:
                    cnt[c] = 1
    frequency=[]
    for key in cnt:
        frequency.append([key,cnt[key]]) #append(frequency to be (frequent items, count) pairs)
    return frequency





def apriori(basket, support, num_baskets) :
    baskets=list(basket)
    #frequent itemsets to store from singles, pairs and triples,etc
    frequentitemsets=[]
    #create empty dic to store singleton count
    c={}
    #empty list to store freq singles/pairs/triples,etc.
    freq=[]
    #define support threshold
    support_threshold=float(len(baskets)/num_baskets)*support
    k=1
    #generating singletons
    for i in baskets:
        for j in i:
            if j in c:
                c[j] += 1
            else:
                c[j] = 1
    
    #count for frequent singles
    for key in c:
        if c[key] >= support_threshold:
            freq.append(key)

    freq.sort()
    #maker singletons to tuple format for future implementation
    freq1=[(x,) for x in freq]
    #append frequent singles to the frequent itemsets
    frequentitemsets=freq1
    k+=1

    #generating pairs
    C=[]
    pair=[]
    for i in itertools.combinations(freq, 2):
        pair=list(i)
        C.append(pair)
    C.sort()
    #Again, make it to tuple form
    C1=[tuple(l) for l in C]
    ck2={}
    for i in baskets:
        for c in C1:
            if (set(c).issubset(i)):
                if c in ck2:
                    ck2[c] += 1
                else:
                    ck2[c] = 1
    #Append frequent pairs to freq 
    freq.clear()
    for key in ck2:
        if (ck2[key] >= support_threshold):
            freq.append(key)
    freq.sort()
    frequentitemsets.extend(freq)
    k+=1

	#countinue to generate triples, quadraples etc in while loop, by increasing k one at a time.
    while len(C)>0 :	
        C.clear()
        for i in range(len(freq)):
            for j in range(i+1, len(freq)): 
                #reference
                list1=list(freq[i])[:k-2]
                list2=list(freq[j])[:k-2]
                if list1==list2: 
                #set union to append elements
                    C.append(sorted(list(set(freq[i]).union(set(freq[j]))))) 
                else:
                    break
        C.sort()
        freq.clear()
        freq=generateFreqitem(baskets, C, support_threshold)
        freq.sort()
        frequentitemsets.extend(freq)
        k+=1


    return frequentitemsets


sc = SparkContext(appName='cz4042_v2')
threshold=0
support=100
input_file=r'./data/yelp_dataset/transactions.csv'
output_file=r'./logs/yelp/yelp_minsup100.txt'


start= time.time()
###read in data file and create baskets
line=sc.textFile(input_file)
header=line.take(1)
rdd=line.filter(lambda s: s not in header).map(lambda s: s.split(",")).map(lambda line: (line[0],line[1])).filter(lambda x: x is not None).distinct()
rdd1=rdd.groupByKey().map(lambda x:(x[0],list(x[1]))).persist()
baskets1=rdd1.map(lambda x: set(x[1])).filter(lambda x: len(x) > threshold)  #baskets to use after filtering threshold
numbaskets=baskets1.count()  #count num of baskets


#son algorithm
phase1reduce=baskets1.mapPartitions(lambda x: apriori(x,support,numbaskets)).map(lambda x: (x,1)).distinct().sortByKey().keys().collect()

phase2reduce=baskets1.mapPartitions(lambda x: son2count_freq(x,phase1reduce)).reduceByKey(lambda x,y: x+y).filter(lambda x: x[1]>=support).sortByKey().map(lambda x:x[0]).collect()

phase1sort=sorted(phase1reduce,key=len)
phase2sort=sorted(phase2reduce,key=len)




#max length of frequent set
o=[]
for i in range(0,len(phase2sort)):
    o.append(len(phase2sort[i]))
    
p=max(o)

#max length of candidate set
o1=[]
for i in range(0,len(phase1sort)):
    o1.append(len(phase1sort[i]))
    
p1=max(o1)

#empty dict to store candidate set by length
g1={}
for i in range(1,p1+1):
    g1.setdefault(i, [])

g={}
for i in range(1,p+1):
    g.setdefault(i, [])

#store candidates set by length into dict
for j in phase1sort:
    g1[len(j)].append(j)

for j in phase2sort:
    g[len(j)].append(j)

time_taken_str = "Duration: " + str(time.time() - start) + "s"

#construct string for candidates
s1=""
for key in g1:
    s1=s1+str(g1[key])[1:-1]+'\n\n'
    s1=s1.replace(',)',')').replace('), (','),(')


s=""
for key in g:
    s=s+str(g[key])[1:-1]+'\n\n'
    s=s.replace(',)',')').replace('), (','),(')

with open(output_file,'w') as f:
    f.write("Candidates:")
    f.write("\n")
    f.write(s1)
    f.write("Frequent Itemsets:")
    f.write("\n")
    f.write(s[:-2]) 
    f.write("\n")
    f.write(time_taken_str)