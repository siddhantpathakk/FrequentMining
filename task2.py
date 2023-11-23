from pyspark import SparkContext
import time

from utils.task2_utils import apriori, son2count_freq

if __name__ == "__main__":
        
    sc = SparkContext(appName='cz4042_task2', master='local[*]')
    
    threshold=0

    support=12500
    
    dataset_name = 'ml-20m'
    input_file=f'./data/{dataset_name}/transactions.csv'
    output_file=f'./logs/{dataset_name}/{dataset_name}_minsup{support}.txt'


    start= time.time()
    
    
    line=sc.textFile(input_file)
    header=line.take(1)
    rdd=line.filter(lambda s: s not in header).map(lambda s: s.split(",")).map(lambda line: (line[0],line[1])).filter(lambda x: x is not None).distinct()
    rdd1=rdd.groupByKey().map(lambda x:(x[0],list(x[1]))).persist()
    baskets1=rdd1.map(lambda x: set(x[1])).filter(lambda x: len(x) > threshold)  #baskets to use after filtering threshold
    numbaskets=baskets1.count()  #count num of baskets


    #son algorithm
    phase1reduce=baskets1.mapPartitions(lambda x: apriori(x,support,numbaskets))\
        .map(lambda x: (x,1)).distinct()\
            .sortByKey().keys().collect()
    
    phase2reduce=baskets1.mapPartitions(lambda x: son2count_freq(x,phase1reduce)).reduceByKey(lambda x,y: x+y)\
        .filter(lambda x: x[1]>=support)\
            .sortByKey().map(lambda x:x[0]).collect()

    phase1sort=sorted(phase1reduce,key=len)
    phase2sort=sorted(phase2reduce,key=len)


    #max length of frequent set
    o, o1 = [], []
    
    for i in range(0,len(phase2sort)):
        o.append(len(phase2sort[i]))
    
    p=max(o)
        
    #max length of candidate set
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
        f.write("Frequent Itemsets:")
        f.write("\n")
        f.write(s[:-2]) 
        f.write("\n\n")
        f.write(time_taken_str)