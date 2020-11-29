import numpy as np
from datasketch import MinHash, MinHashLSH
import queue as Q
import sys
import time
per = 64
lsh_thres = 0.2
#lsh_thres = (float)(sys.argv[2])#0.3
print("lsh_thres", lsh_thres)
thres = 64

dataset = sys.argv[1]

configline = open("../data/" + dataset + ".config").readline()
numv = (int)(configline.split(" ")[0])
nume = (int)(configline.split(" ")[1])
print(numv, nume)
f = open("../data/" + dataset + ".graph")
ptr = f.readline().strip("\n").strip(" ").split(" ")
idx = f.readline().strip("\n").strip(" ").split(' ')
if len(idx) != nume:
    print("error idx", len(idx))
    exit()
if len(ptr) != numv + 1:
    print("error ptr",len(ptr))
    exit()

t0 = time.time()
lsh = MinHashLSH(threshold=lsh_thres, num_perm=per)
allver = []
lists = [[] for i in range(numv)]
for i in range(numv):
    m = MinHash(num_perm=per)
    for iter in range((int)(ptr[i]), (int)(ptr[i+1])):
        m.update(str(idx[iter]).encode('utf-8'))
        lists[i].append(idx[iter])
    lsh.insert(str(i), m)
    allver.append(m)
#res = lsh.query(allver[0])
#print(res)
t1 = time.time()
print("init LSH", t1 - t0)

def jd(l1,l2):
    if len(l1) == 0 or len(l2) == 0:
        return 0
    s1 = set(l1)
    s2 = set(l2)
    return (float)(len(s1.intersection(s2))) / len(s1.union(s2))
#for item in res:
#    print(jd(lists[0], lists[(int)(item)]))
#print(lists[0])
#print(lists[(int)(res[-1])])
# print(lists)

class Pair(object):
    def __init__(self,p1,p2,similarity):
        self.p1 = p1
        self.p2 = p2
        self.simi = similarity
    def __lt__(self,other):#operator < 
        return self.simi > other.simi
    def __str__(self):
        return str(self.p1) + ' ' + str(self.p2) + ' ' + str(self.simi)
        # return '(' + str(self.priority)+',\'' + self.description + '\')'

que = Q.PriorityQueue()


# goodpairs = [[] for i in range(numv)]
sset = set()
def makenum(a, b):
    if a > b:
        tmp = a
        a = b
        b = tmp
    return a * numv + b
t2 = time.time()
for i in range(numv):
    if i % 100000 == 0:
        print("reach", i)

    if ptr[i] == ptr[i + 1]:
        continue
    res = lsh.query(allver[i])
    # goodpairs[i] = res
    # print(len(res))
    # print(ptr[i], ptr[i+1])
    for item in res:
        if (int)(item) == i or makenum(i, (int)(item)) in sset:
            continue
        que.put(Pair(i, (int)(item), jd(lists[i], lists[(int)(item)])))
        sset.add(makenum(i, (int)(item)))
        # print("add", i, (int)(item))
print("queuesize:", que.qsize())
t3 = time.time()
print("query LSH", t3 - t2)
# cnt = 0
# while not que.empty():
#     item = que.get()
#     print(item)
#     # print (que.get())
#     print(lists[item.p1])
#     print("second", lists[item.p2])
#     cnt += 1
#     if cnt > 100:
#         break
cluster_id = [i for i in range(numv)]
cluster_sz = [1 for i in range(numv)]
deleted = [0 for i in range(numv)]

def root(i):
    while i != cluster_id[i]:
        cluster_id[i] = cluster_id[cluster_id[i]]
        i = cluster_id[i]
    return i

num_cluster = numv

t4 = time.time()
while (not que.empty()) and num_cluster > 0:
    item = que.get()
    p1 = item.p1
    p2 = item.p2
    # print(p1, p2)
    sset.remove(makenum(p1, p2))
    if p1 == cluster_id[p1] and p2 == cluster_id[p2]:
        if deleted[p1] or deleted[p2]:
            continue
        if cluster_sz[p1] < cluster_sz[p2]:
            cluster_id[p1] = p2
            num_cluster = num_cluster - 1
            cluster_sz[p2] = cluster_sz[p1] + cluster_sz[p2]
            if cluster_sz[p2] >= thres:
                deleted[p2] = 1
                num_cluster = num_cluster - 1
        else:
            cluster_id[p2] = p1
            num_cluster = num_cluster - 1
            cluster_sz[p1] = cluster_sz[p1] + cluster_sz[p2]
            if cluster_sz[p1] >= thres:
                deleted[p1] = 1
                num_cluster = num_cluster - 1
    else:
        p1 = root(p1)
        p2 = root(p2)
        if deleted[p1] or deleted[p2]:
            continue
        if p1 != p2 and not makenum(p1, p2) in sset:
            que.put(Pair(p1, p2, jd(lists[p1], lists[p2])))
            sset.add(makenum(p1, p2))
            # print("add", i, (int)(item))
t5 = time.time()
print("clustering", t5 - t4)
print("ffff2")
clusters = {}
t6 = time.time()
for i in range(numv):
    ro = root(i)
    if ro in clusters:
        clusters[ro].append(i)
    else:
        clusters[ro] = [i]
print("ffff")
print(len(clusters))
t7 = time.time()
print("put into clusters", t7 - t6)
with open("../data/" + dataset + ".new_reorder_thres_" + str(lsh_thres), 'w') as f:
    for k in clusters:
        for item in clusters[k]:
            f.write(str(item) + ' ')
print("write back", time.time() - t7)
