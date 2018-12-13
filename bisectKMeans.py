import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import sys

def bisect(oldnum, newnum, tfidfMat, clusterList, iterNum):
    docNum = tfidfMat.shape[0]
    bestAvgSim = 0
    for j in range(iterNum):
        changed = 1
        #print "kmeans iteration", j
        tmpList = clusterList[:]
        #pick starting centroids as random vectors within old cluster
        while (1):
            centIndex1 = random.randrange(docNum)
            if tmpList[centIndex1] == oldnum:
                break
        while (1):
            centIndex2 = random.randrange(docNum)
            if tmpList[centIndex2] == oldnum:
                break

        startCent1 = tfidfMat.getrow(centIndex1)
        startCent2 = tfidfMat.getrow(centIndex2)
        #cosine similarity matrix measured against cluster centroid
        centSim1 = cosine_similarity(tfidfMat, startCent1)
        centSim2 = cosine_similarity(tfidfMat, startCent2)
        for i in range(docNum):
            if (clusterList[i] == oldnum and centSim2[i][0] >= centSim1[i][0]):
                tmpList[i] = newnum
        while (changed != 0):
            # calculate new centroid of each cluster
            changed = 0
            newCent1 = centroid(tmpList, tfidfMat, oldnum)
            newCent2 = centroid(tmpList, tfidfMat, newnum)
            centSim1 = cosine_similarity(tfidfMat, newCent1)
            centSim2 = cosine_similarity(tfidfMat, newCent2)
            for i in range(docNum):
                prv = tmpList[i]
                if (clusterList[i] == oldnum and centSim1[i][0] >= centSim2[i][0]):
                    tmpList[i] = oldnum
                elif (clusterList[i] == oldnum and centSim1[i][0] < centSim2[i][0]):
                    tmpList[i] = newnum
                if prv != tmpList[i]:
                    changed += 1
        avgsim1 = (newCent1.multiply(newCent1).sum(1)).item(0)
        avgsim2 = (newCent2.multiply(newCent2).sum(1)).item(0)
        if ((avgsim1 + avgsim2) > bestAvgSim):
            bestList = tmpList[:]
        bestAvgSim = avgsim1 + avgsim2
    return bestList


def centroid(clusterlist, tfidfMatrix, clusterNum):
    newCent, ccount = 0, 0
    for i in range(len(clusterlist)):
        if (clusterlist[i] == clusterNum):
            newCent += tfidfMatrix[i]
            ccount += 1
    return (newCent / ccount)

def largest_cluster_num(clusterlist):
    return max(clusterlist, key=clusterlist.count)

def avg_clus_sim(clusterlist, tfidfMatrix, num):
    sim = 0
    for i in range(1, num + 1):
        cent = centroid(clusterlist, tfidfMatrix, i)
        sim += (cent.multiply(cent).sum(1)).item(0)
    return sim/num

f = open(sys.argv[1], "r")
docs = f.read().splitlines()
f.close()
vectorizer = TfidfVectorizer()
tfidfMat = vectorizer.fit_transform(docs)
clusters = [1 for x in range(tfidfMat.shape[0])]
for i in range(1, int(sys.argv[3])):
    #print "bisecting iteration", i
    largest = largest_cluster_num(clusters)
    nextCluster = i + 1
	#try 5 iterations of kmeans 
    clusters = bisect(largest, nextCluster, tfidfMat, clusters, 5)
    avgsim = avg_clus_sim(clusters, tfidfMat, nextCluster)
    print avgsim

w = open(sys.argv[2], "w")
for i in range(len(clusters)):
    w.write(str(clusters[i]) + "\n")

w.close()
