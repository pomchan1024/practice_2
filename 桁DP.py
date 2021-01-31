# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:11:35 2021

@author: hyakk
"""

import numpy as np
from collections import deque
import sys
sys.setrecursionlimit(20000000)

#桁DP
N=list(input())
D=len(N)
K=int(input())

def dp(d,smaller,seen):#上位d桁目までの数字で、Nに一致しているか→smaller、何回0が出たか→seen
    if d==0 and smaller==0 and seen==0: 
        return 1
    if d==0 and smaller==1: 
        return 0
    if d==0 and seen>0: 
        return 0
    x=int(N[d-1])
    if smaller==0:
        if x==0:
            if seen>0:
                return dp(d-1,0,seen-1)
            if seen==0:
                return 0
        else: 
            return dp(d-1,0,seen)
    l=[]
    if smaller==1:
        l.append(dp(d-1,1,seen)*9)
        if seen>0:
            l.append(dp(d-1,1,seen-1))
        if x==0:
            if seen==0:
                return sum(l)
            else:
                return sum(l)+dp(d-1,0,seen-1)
        else:
            if seen==0:
                return sum(l)+dp(d-1,0,seen)*(x-1)
            else:
                return sum(l)#dp(d-1,0,seen-1)

print (dp(D,1,K))



"""
S=list(input())
#print(S)

def dp(i,j):#i回目p、j回gを出したとする。
    if i==0 and j==0:
        return 0
    if i>j:
        return -1000
    #print (i,j)
    if S[i+j-1]=='p':
        if i>0 and j>0:
            return max(dp(i,j-1)-1,dp(i-1,j))
        if i==0 and j>0:
            return dp(i,j-1)-1
    else:
        if i>0 and j>0:
            return max(dp(i,j-1),dp(i-1,j)+1)
        if i==0 and j>0:
            return dp(i,j-1)
    

print (max(dp(i,len(S)-i) for i in range(len(S)+1)))
"""


"""
K=int(input())
V=list(map(int,input().split()))
N=len(V)
G=[[] for i in range(N)]
for i in range(N-1):
    u,v=map(int,input().split())
    G[u-1].append(v-1)

todo=deque()
seen=[0]*N
sumlist=[0]*N
def bfs(todo,sumlist):
    while len(todo)>0:
        x=todo.pop()
        if sumlist[x]>K:
            continue
        for i in G[x]:
            sumlist[i]=sumlist[x]+V[i]
            #print (sumlist[i])
            if sumlist[i]==K:
                print("Yes")
                return
            todo.append(i)
            
for i in range(N):
    todo=deque()
    sumlist=[0]*N
    sumlist[i]=V[i]
    todo.append(i)
    bfs(todo,sumlist)
             
"""   
    
    

"""
V=list(map(int,input().split()))
N=len(V)
G=[[] for i in range(N)]
for i in range(N-1):
    u,v=map(int,input().split())
    G[u-1].append(v-1)
    #G[v-1].append(u-1)
print (G)
seen=[-1]*N
distance=[0]*N
todo=deque()

def dfs(todo,seen,distance):
    while len(todo)>0:
        x=todo.popleft()
        if seen[x]>-1:
            continue
        seen[x]=0
        for i in G[x]:
            if seen[i]==-1:
                todo.append(i)
                distance[i]=distance[x]+1

todo.append(0)
dfs(todo,seen,distance)
print (distance)
"""


"""
R,C=map(int,input().split())
sy,sx=map(int,input().split())
gy,gx=map(int,input().split())

maplist=[list(input()) for i in range(R)]

todo=deque()
todo.append((sy-1,sx-1))

seen=[[-1]*C for i in range(R)]#-1未知、0既知
distance=[[0]*C for i in range(R)]
l=[]
def bfs(todo,seen,distance):
    while len(todo)>0:
        y,x=todo.popleft()
        if seen[y][x]>-1:
            continue
        seen[y][x]=0
        for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
            nx=x+dx
            ny=y+dy
            if nx<0 or ny<0 or nx>=C or ny>=R:
                continue
            if maplist[ny][nx]=='#':
                continue
            todo.append((ny,nx))
            distance[ny][nx]=distance[y][x]+1

bfs(todo,seen,distance)
print (distance[gy-1][gx-1])
"""

"""
V=list(map(int,input().split()))
N=len(V)
G=[[] for i in range(N)]
for i in range(N-1):
    u,v=map(int,input().split())
    G[u-1].append(v-1)
    #G[v-1].append(u-1)
print (G)
seen=[-1]*N
todo=deque()

def dfs(todo,seen):
    while len(todo)>0:
        x=todo.pop()
        if seen[x]>-1:
            continue
        seen[x]=0
        for i in G[x]:
            if V[i]>=V[x]:
                todo.append(i)
            else:
                print ("No")
                break
todo.append(0)
dfs(todo,seen)
print ("Yes")
"""


"""
N = int(input())
G = [[] for i in range(N)]#先に隣の点と、その距離をまとめておく
for i in range(N-1):
    u, v, w = map(int, input().split())
    G[u-1].append([v-1, w])
    G[v-1].append([u-1, w])

print (G)

ans=[-1]*N
def dfs(node,distance):
    for next_node,edge_size in G[node]:
        if ans[next_node]==-1:
            ans[next_node]=((distance+edge_size)%2)
            dfs(next_node,distance+edge_size)
"""

"""
N=int(input())
V=np.arange(N)
s,t=map(int,input().split())
To=[list(map(int,input().split())) for i in range(N)]

seen=[0]*N
todo=deque()

def dfs(todo,seen):
    while len(todo)>0:
        x=todo.pop()
        if seen[x]==0:
            seen[x]=1
        else:
            continue
        if len(To[x])==0:
            continue
        else:
            for i in To[x]:
                todo.append(i)
todo.append(s)
dfs(todo,seen)
print(seen[t])
"""

"""
N,K=map(int,input().split())
D=list(map(int,input().split()))
T=[0,1,2,3,4,5,6,7,8,9]
for i in range(K):
    T.remove(D[i])
#print (T)

l=[]
def dfs(n):
    if n>=N:
        l.append(n)
        return
    for i in range(len(T)):
        if i==0 and n==0 and T[0]==0 : 
            continue
        dfs(n*10+T[i])

dfs(0)
print (min(l))
"""


"""
N=int(input())

l=[]
def dfs(n):
    if n>10**N:
        return
    if n>=10**(N-1):
        l.append(n)
    dfs(n*10+1)
    dfs(n*10+2)
    dfs(n*10+3)

dfs(0)

for i in range(len(l)):
    a=str(l[i])
    print(a.replace('1','a',N).replace('2','b',N).replace('3','c',N))
"""

"""
S=list(input())
T=list(input())
def dp(s,t):#左からs個め、t個めの場合の編集距離
    if s==0 or t==0:
        return max(s,t)
    if S[s-1]==T[t-1]:
        return dp(s-1,t-1)
    else:
        return min(dp(s-1,t)+1,dp(s,t-1)+1,dp(s-1,t-1)+1)

print(dp(len(S),len(T)))
"""

"""
S=list(input())
T=list(input())

def dp(s,t):
    if s==0 or t==0:
        return 0
    if S[s-1]==T[t-1]:
        return dp(s-1,t-1)+1
    else:
        return max(dp(s,t-1),dp(s-1,t))
print (dp(len(S),len(T)))
"""

"""
N,A,K=map(int, input().split())
alist=list(map(int,input().split()))

def dp(n,a):#n個目までの数字を使って、総和をaできる場合の最小個数
    if a>0:
        if n==0:
            return 1000
    elif a==0:
        if n==0:
            return 0
        else:
            return 1000
    if alist[n-1]<=a:
        return min(dp(n-1,a-alist[n-1])+1,dp(n-1,a))
    else:
        return dp(n-1,a)
if dp(N,A)<=K:
    print ("Yes")
else:
    print ("No")
"""

"""
N,A=map(int, input().split())
alist=list(map(int,input().split()))

def dp(n,a):#n個目までの整数から選んで、総和をaにできる場合の数
    if n==0 and a>0:
            return 0
    if a==0:
            return 1
    if alist[n-1]<=a:
        return dp(n-1,a-alist[n-1])+dp(n-1,a)
    else:
        return dp(n-1,a)
print (dp(N,A))

"""
"""
N,W=map(int, input().split())
A=[list(map(int, input().split())) for i in range(N)]
#A[i][1]:i番目の商品の価値

def dp(n,w): #n個目までの商品を使って、重さがwの場合の価値最大値
    if n==0 or w==0:
        return 0
    if n==1:
        if A[0][0]==w:
            return A[0][1]
        else:
            return 0
    if w>=A[n-1][0]:
        return max(dp(n-1,w-A[n-1][0])+A[n-1][1],dp(n-1,w))
    else:
        return dp(n-1,w)
print (dp(N,W))
        
"""


"""

N,A=map(int, input().split())
alist=list(map(int,input().split()))

def dp(n,a):#n個目までの整数から選んで、総和をaにできるなら1、できないなら0
    if n==0 and a>0:
            return 0
    if a==0:
            return 1
    if alist[n-1]<=a:
        return max(dp(n-1,a-alist[n-1]),dp(n-1,a))
    else:
        return dp(n-1,a)
print (dp(N,A))
"""


"""
N=int(input())
A=[list(map(int, input().split())) for i in range (N)]
#A[i][0]i日目の、海で泳ぐ幸福度
def dp(n,index):
    if n==0:
        return 0
    if index==0:
        return max(dp(n-1,1)+A[n-1][0],dp(n-1,2)+A[n-1][0])
    elif index==1:
        return max(dp(n-1,0)+A[n-1][1],dp(n-1,2)+A[n-1][1])
    else:
        return max(dp(n-1,0)+A[n-1][2],dp(n-1,1)+A[n-1][2])

print (max(dp(N,0),dp(N,1),dp(N,2)))
"""


"""
seen=[0,0,0]#7,5,3を使用したら1に変換
K=1000

l=[]
def dp(k,seen):
    if k>K:
        return
    if sum(seen)==3:
        l.append(1)
    dp(10*k+7, [1,seen[1],seen[2]])
    dp(10*k+5, [seen[0],1,seen[2]])    
    dp(10*k+3, [seen[0],seen[1],1])
    #print (k)
dp(0,seen)
print(sum(l))

"""
"""
from collections import deque

R, C=map(int,input().split())

maplist=[]
wlist=[]
for i in range(R):
    a=list(map(int,input().split()))
    #print (a)
    wlist.append([i for i, x in enumerate(a) if x == 0])
    maplist.append(a)
#print (wlist[0])

pondsize=[]#各池のサイズを格納
todo=deque()

seen=[[-1]*C for i in range(R)]

def bfs(todo,seen):
    s=0
    while len(todo)>0:
        #print (todo)
        x,y=todo.popleft()
        if seen[y][x]==-1:
            s=1
        for dx,dy in((1,0),(0,1),(-1,0),(0,-1)):
            nx=x+dx
            ny=y+dy
            if nx<0 or ny<0 or nx>=C or ny>=R:
                continue
            if seen[ny][nx]==0:
                continue
            #print(nx,ny)
            seen[ny][nx]=0
            if maplist[ny][nx]==0:
                s+=1         
                todo.append((nx,ny))
    pondsize.append(s)
for i in range(R):
    if len(wlist[i])==0:
        continue
    print (wlist[i])
    for j in range(len(wlist[i])):
        if seen[i][wlist[i][j]]==0:
            continue
        todo.append((wlist[i][j],i))
        print (todo)
        bfs(todo,seen)

print (pondsize)
"""

"""
R, C=map(int,input().split())
sy,sx=map(int,input().split())
gy,gx=map(int,input().split())
maplist=[list(input()) for i in range (R)]

d=0
todo=deque()
todo.append((sx-1,sy-1))

dist=[[-1]*C for i in range(R)]

dist[sy-1][sx-1]=0

def bfs(todo, dist):
    while len(todo)>0:
        print (todo)
        x,y=todo.popleft()
        
        d=dist[y][x]
        for dx,dy in ((1,0),(0,1),(-1,0),(0,-1)):
            nx=x+dx
            ny=y+dy
            if nx<0 or ny<0 or nx>=C or ny>=R:
                continue
            if maplist[ny][nx]=='#':
                continue
            if dist[ny][nx]==-1:
                dist[ny][nx]=d+1
                todo.append((nx,ny))

bfs(todo,dist)
print (dist[gy-1][gx-1])

"""


"""
H,W =map(int, input().split())
maplist=[input() for i in range(H)]

dist=[[-1]*W for i in range(H)]
black_cell=deque()

for h in range(H):
    for w in range(H):
        if maplist[h][w]=='#':
            black_cell.append((h,w))
            dist[h][w]=0
def bfs(black_cell,dist):
    d=0
    while len(black_cell)>0:
        h,w=black_cell.popleft()
        d=dist[h][w]
        for dy,dx in((1,0),(0,1),(-1,0),(0,-1)):
            nh=h+dy
            nw=w+dx
            if nh<0 or nw<0 or nh>=H or nw>=W:
                continue
            if dist[nh][nw]==-1:
                dist[nh][nw]=d+1
                black_cell.append((nh,nw))
    return d

print(bfs(black_cell,dist))

"""


"""

A=list(map(int,input().split()))
n=len(A)

def dp(i):#i個目までの数列で、和が最大になるものの値
    if i==0:
        return 0
    #dp(i-1)か、それともA[i-1]を含む数列和のいずれが大きいか
    l=[]
    for j in range(i):
       l.append(sum(A[j:i]))
       print (l)
    nmax=max(l)
    return max(dp(i-1),nmax)

print (dp(n))
"""
"""
C=list(input().split())
H=list(input().split())

def HBcounter(x,y):
    Hans=0
    Bans=0
    l=[]
    for i in range(len(C)):
        if x[i]==y[i]:
            Hans+=1
            l.append(x[i])
    sc=set(C)
    sh=set(H)
    print(l)
    for j in range(len(l)):
        sc.remove(l[j])
        sh.remove(l[j])
        print(sc,sh)
    for k in sh:
        if k in sc:
            Bans+=1
    return [Hans,Bans]
print (HBcounter(C,H))
"""

"""
n,c=map(int,input().split())
A=[list(map(int,input().split())) for i in range(n)]

def dp(x):#x日終了時までの最小の合計
    if x==0:
        return 0
    l=[]
    for i in range(n):
        if A[i][0]<=x and A[i][1]>=x:
            l.append(A[i][2])
    return dp(x-1)+min(sum(l),c)

print (dp(max(A[:][1])))
            
"""

"""
n=int(input())
#25a+10b+5c+d=n
coins=[25,10,5,1]
def dp(x,coins,index):#iこのコインで、和がxに一致するような場合の数
    if index>=len(coins)-1:#コインが1なら場合の数1
        return 1
    denocoin=coins[index]
    u=int(x/denocoin)
    l=[]
    for i in range(u+1):
       print (x-i*denocoin)
       l.append(dp(x-i*denocoin,coins,index+1))
    return sum(l)

print (dp(n,coins,0))
#print(sum(dp(k,n) for k in range(1,n+1)))
"""


"""
#n段階段の上り方(by 1,2,3)
n=int(input())

def dp(x):#xは階段数
    if x==1:
        return 1
    if x==2:
        return 2
    if x==3:
        return 4
    return dp(x-3)+dp(x-2)+dp(x-1)

print (dp(n))

"""