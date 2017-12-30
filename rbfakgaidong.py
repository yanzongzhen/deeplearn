#coding:utf-8
from numpy import *
import matplotlib.pyplot as plt

def dataset():
    train_x=[-3.6606,   -3.4284, 0.1732,   -3.2262,    2.5452,    2.5404,    1.7795,   -2.8011,    1.2768,    0.1488,
    3.7838,    1.1919,    2.4026,   -0.3696,   -0.5409,    2.6025,   -3.3322,   -2.9346,   -2.6129,   -0.8725,
    2.6510,    2.4269,   -3.5162,   -0.8059,    0.2150,   -0.6656,    1.2549,    1.0238,   -1.6641,   -0.5468,
   -3.8761,    3.8725,   -2.6627,   -3.1503,   -1.0207,   -2.4151,   -0.0825,   -1.2841,    3.6130,    3.3627,
   -3.5786,    1.9029,   -1.8470,   -0.6173,    0.3830,    3.5419,   -0.6580,    3.8644,   -1.5884,    1.6088,
    1.3307,    0.3130,    1.5848,    1.3322,   -2.5749,   -2.9759,    3.9926,   -2.6310,   -3.7392,    0.4896,
    3.0549,    1.3534,   -2.4765,   -1.0487,   -0.3142,    3.8531,   -2.7488,    2.8442,    1.1581,   -0.9898,
   -2.4726,   -0.5740,   -0.1438,   -3.0351,    0.7161,   -2.1905,   -0.9230,    0.6639,   -1.9856,   -1.6765,
    0.9367,   -1.8778,    2.5950,    3.8613,    1.8420,   -1.2490,    0.6726,   -3.1378,    3.2505,    3.0372,
    2.5421,   -1.9142,    0.7549,   -3.8199,   -0.5979,   -1.4982,   -2.7081,   -2.5699,   -0.6169,   -3.2462]
   
    train_y=[0.0216,    0.1487,    0.9793,    0.0484,    0.5870,    0.5269,    1.2677,    0.4757,    1.4787,    0.8801,
    0.0059,    1.4177,    0.5691,    1.8560,    1.9327,    0.3960,    0.0421,    0.1965,    0.6061,    2.5249,
    0.5593,    0.5742,   -0.0399,    2.6287,    1.0666,    2.2262,    1.2983,    1.3054,    2.2439,    2.0592,
   -0.0051,    0.0609,    0.6059,    0.0597,    2.5869,    0.8239,    1.1508,    2.6602,    0.0391,   -0.2248,
    0.0093,    1.2649,    1.8254,    2.2567,    0.9656,    0.0439,    2.2542,   -0.1395,    2.3701,    1.5378,
    1.4669,    0.9289,    1.3173,    1.4541,    0.6960,    0.3275,   -0.0263,    0.5798,    0.2355,    0.7400,
    0.3948,    1.4908,    0.9066,    2.5300,    1.5237,   -0.0102,    0.5168,    0.1092,    1.4672,    2.5405,
    0.8189,    2.1484,    1.3230,    0.3551,    1.2152,    1.2120,    2.6315,    0.9800,    1.5334,    2.3314,
    1.2897,    1.8681,    0.5416,    0.0766,    1.2337,    2.8323,   1.1740,    0.2147,    0.0364,    0.1141,
    0.6139,    1.6425,    1.1432,   -0.1695,    2.2298,    2.5882,    0.5167,    0.6722,    1.9140,    0.2016]

    test_x=[-4.0000,   -3.9200,   -3.8400,   -3.7600,   -3.6800,   -3.6000,   -3.5200,   -3.4400,   -3.3600,   -3.2800,
   -3.2000,   -3.1200,   -3.0400,   -2.9600,   -2.8800,   -2.8000,   -2.7200,   -2.6400,   -2.5600,   -2.4800,
   -2.4000,   -2.3200,   -2.2400,   -2.1600,   -2.0800,   -2.0000,   -1.9200,   -1.8400,   -1.7600,   -1.6800,
   -1.6000,   -1.5200,   -1.4400,   -1.3600,   -1.2800,   -1.2000,   -1.1200,   -1.0400,   -0.9600,   -0.8800,
   -0.8000,   -0.7200,   -0.6400,   -0.5600,   -0.4800,   -0.4000,   -0.3200,   -0.2400,   -0.1600,   -0.0800,
         0,   0.0800,   0.1600,    0.2400,    0.3200,    0.4000,    0.4800,    0.5600,    0.6400,    0.7200,
    0.8000,   0.8800,    0.9600,    1.0400,    1.1200,    1.2000,    1.2800,    1.3600,    1.4400,    1.5200,
    1.6000,   1.6800,    1.7600,    1.8400,    1.9200,    2.0000,    2.0800,    2.1600,    2.2400,    2.3200,
    2.4000,   2.4800,    2.5600,   2.6400,    2.7200,    2.8000,    2.8800,    2.9600,    3.0400,    3.1200,
    3.2000,    3.2800,    3.3600,    3.4400,    3.5200,    3.6000,    3.6800,    3.7600,    3.8400,    3.9200,
    4.0000]
    test_y=[
    0.0137,    0.0181,    0.0237,    0.0309,    0.0401,    0.0515,    0.0657,    0.0833,    0.1048,    0.1309,
    0.1622,    0.1997,    0.2439,    0.2958,    0.3559,    0.4252,    0.5040,    0.5929,    0.6921,    0.8016,
    0.9213,    1.0504,    1.1882,    1.3331,    1.4836,    1.6376,    1.7924,    1.9453,    2.0933,    2.2330,
    2.3611,    2.4742,    2.5693,    2.6435,    2.6943,    2.7200,    2.7194,    2.6922,    2.6389,    2.5608,
    2.4602,    2.3401,    2.2042,    2.0568,    1.9026,    1.7465,    1.5936,    1.4484,    1.3154,    1.1982,
    1.1000,    1.0228,    0.9679,    0.9354,    0.9247,    0.9342,    0.9615,    1.0036,    1.0569,    1.1177,
    1.1822,    1.2463,    1.3067,    1.3599,    1.4034,    1.4349,    1.4530,    1.4568,    1.4460,    1.4209,
    1.3824,    1.3317,    1.2705,    1.2005,    1.1237,    1.0421,    0.9576,    0.8721,    0.7872,    0.7044,
    0.6249,    0.5497,    0.4795,    0.4148,    0.3559,    0.3029,    0.2558,    0.2143,    0.1781,    0.1468,
    0.1202,    0.0976,    0.0786,   0.0629,    0.0499,    0.0393,    0.0308,    0.0239,    0.0184,    0.0141,
    0.0107]
    return train_x,train_y,test_x,test_y
def creatCent(data,k):
    InitialCluster =[]
    for i in range(k):
        InitialCluster.append(data[i])
    """
    输入：数据集, 聚类个数
    输出：k个随机质心的矩阵
    """
    return InitialCluster
def distEclud(vecA, vecB):
    """
    输入：向量A, 向量B
    输出：两个向量的欧式距离
    """
    value = sqrt(sum(power(vecA[0] - vecB[0], 2))) 
    return value
    
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:j]) 
        rangeJ = float(max(dataSet[:j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids


#进行kmean聚类划分
def kmean(data,k,distMeans=distEclud, createCent=randCent):
        numSamples = shape(data)[0]
        """
            输入：数据集, 聚类个数, 距离计算函数, 生成随机质心函数
            输出：质心矩阵, 簇分配和距离矩阵
        """
        clusterAssment = mat(zeros((numSamples,10)))
        centroids = creatCent(data,k)
        clusterChanged = True
        while clusterChanged:
            clusterChanged = False
            for i in range(numSamples):
                minDist = inf
                minIndex = -1
                for j in range(k):
                    #distJI = distEclud(centroids[j,:],data[i,:])
                    print('****%s#######%s'%(centroids,data))
                    distJI = distMeans(centroids[j:],data[i:])
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                    if clusterAssment[i,0] != minIndex:
                        clusterChanged = True
                    clusterAssment[i,:] = minIndex,minDist**2
                for cent in range(k):
                    ptsInClust = data[nonzero(clusterAssment[:,0].A == cent)[0]]
                    centroids[cent,:] = mean(ptsInClust,axis = 0)
        return centroid,clusterAssmen
        

#求得高斯核函数
def Gauss(means,train,mindist):
    Data = tile(train, (10, 1))
    Data=Data.T
    for i in range(10):
        for j in range(train.shape[0]):
            Data[j,i] = 1/(1+exp( (Data[j,i]-means[i])*(Data[j,i]-means[i]) / (2*(mindist[i])**2)))
    return Data

def main():
    train_x,train_y,test_x,test_y=dataset()
    M=10
    train_x=array(train_x)
    dataNums=train_x.shape[0]
    #index代表数据划分聚类，means为是个聚类中心值大小
    index,means=kmean(train_x,M)

    #distance为其各个聚类中心值的差值
    distance=zeros((M,M))
    for i in range(M):
        for j in range(M):
            distance[i,j]=abs(means[i]-means[j])
            if distance[i,j]==0:
                distance[i,j]=100
    #Excon为获取各个聚类中心距离的最小值
    mindist=[];
    for i in range(M):
        mindist.append(min(distance[:,i]))
    
    print("%d个聚类均值: %s" %(M,means))
    print("%d个聚类方差：%s" %(M,mindist))


    #通过高斯径向基函数的输出
    Data=Gauss(means,train_x,mindist) #经过高斯激活函数后的值 
    b=ones((100,1))#b值
    Data=column_stack((Data,b))#将b值添加进输入矩阵中  
    w1=dot(Data.T,Data)
    W=linalg.inv(w1)#求得逆矩阵
    Weight=dot(W,Data.T)
    weight=dot(Weight,train_y)#求得权重
    print("十一个权值w：",weight)

    #对测试样本进行处理
    test_x=array(test_x)
    B=ones((test_x.shape[0],1))
    testData=Gauss(means,test_x,mindist) 
    testData=column_stack((testData,B))
    TestY=dot(weight,testData.T)

    #分别对测试输出以及测试数据进行显示
    f1=plt.figure(1)
    #用点表示两个拟合
    #plt.scatter(arange(101),test_y,color='b')
    #plt.scatter(arange(101),TestY,color='g')
    #用曲线表示两个拟合
    x=arange(0,101)
    plt.plot(x,TestY,color='g')
    plt.plot(x,test_y,color='r')
    plt.show()

main()
