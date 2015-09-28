#!/usr/bin/env python
#-*- coding:utf-8 -*-
from numpy import *
import re
# import chardet
import multiprocessing
from multiprocessing import Pool  #多进程
import time
import datetime
import pymysql
import jieba
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

#from math import *
"""
@author: Wicle Qian
2015-07-13
Nature Language Processing：Naive Bayes to classify text
classify 1:negative
classify 0:positive
"""

################词表到向量的转换函数#########################

def loadData(fileName):
    try:
        #trainList=open(fileName).read()
        fr = open(fileName).readlines()
    except:
        print "打开文件异常"
        return 0
    pos=[];classVec=[]
    for line in fr:
        pos.append(line.decode('gbk','ignore'))
        classVec.append(0)   #0是正常
    print pos[0],pos[1]#.encode('utf-8')
    print classVec

#####    提取训练数据    #####
def testTextParse(filename,classify):
    text = open(filename).read()
    pattern = '<text>(.*?)</text>'
    str_list = re.findall(pattern, text, re.S)  #re.S :多行匹配
    doc_list = []
    ptn = re.compile('\\s*')#\s是指空白,包括空格、换行、tab缩进等所有的空白, 可以把正则表达式编译成一个正则表达式对象。可以把那些经常使用的正则表达式编译成正则表达式对象，这样可以提高一定的效率。

    for doc in str_list:
        doc = ptn.split(doc)
        doc_list.append([term for term in doc if len(term)>=1 and term != ','and term != '.'and term != '!'and term != '?'and term != '('and term != ')'
                         and term != '\"'and term != '\''
                         and term != '\xa1\xa3' and term != '\xa3\xac' and term != '\xa3\xbf'and term != '\xa3\xa1'and term != '\xa3\xbb' #'\xa1\xa3':。  '\xa3\xac'：，'\xa3\xbf'：？ '\xa3\xbb' ：分号
                         and term != '\xa3\xba'and term != '\xa1\xb0'and term != '\xa1\xb1'and term != '\xa1\xae'and term != '\xa1\xaf'
                         and term != '\xa3\xa8'and term != '\xa3\xa9'and term != '\xa1\xa2'
                         ])
    if classify==0:
        classVec=zeros( len(doc_list))
    else :
        classVec=ones(len(doc_list))
    print 'class',classify,':len of doc_list',len(doc_list),' ,len of classVec',len(classVec)
    #print classVec
    return doc_list,classVec

#####    提取测试数据    #####
def testText(filename):
    if ".txt" not in filename:  # judge whether the filename end with .txt
        print filename," is not a file"
        return 0
    try:
        open(filename,'r')
    except:
        print "no file name:",filename
        return 0
    text = open(filename).read()
    pattern = '<text>(.*?)</text>'
    str_list = re.findall(pattern, text, re.S)  #re.S :多行匹配
    doc_list = []
    ptn = re.compile('\\s*')

    for doc in str_list:
        doc = ptn.split(doc)
        doc_list.append([term for term in doc if len(term)>=1 and term != ','and term != '.'and term != '!'and term != '?'and term != '('and term != ')'
                         and term != '\"'and term != '\''
                         and term != '\xa1\xa3' and term != '\xa3\xac' and term != '\xa3\xbf'and term != '\xa3\xa1'and term != '\xa3\xbb' #'\xa1\xa3':。  '\xa3\xac'：，'\xa3\xbf'：？ '\xa3\xbb' ：分号
                         and term != '\xa3\xba'and term != '\xa1\xb0'and term != '\xa1\xb1'and term != '\xa1\xae'and term != '\xa1\xaf'
                         and term != '\xa3\xa8'and term != '\xa3\xa9'and term != '\xa1\xa2'
                         ])
        print doc_list
    print len(doc_list)
    for i in range(len(doc_list[0])):
            print doc_list[0][i].decode('utf-8'),
    return doc_list


#创建一个包含在所有文档中出现的不重复词的列表，使用set数据类型，将词条列表输给set构造函数，就会返回一个不重复表
def createVocabList(dataSet):
    vocabSet=set([])     #创建一个空集
    for document in dataSet:
        vocabSet=vocabSet | set(document )       #创建2个集合的并集
    return list(vocabSet)

########  词集模型  ##########
#将每个词的出现与否作为一个特征，这称为词集模型(set-of-words model)
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print "the word: %s is not in my Vocabulary!"% word
    return returnVec


################朴素贝叶斯分类器训练函数#################
##########   运用稀疏矩阵的朴素贝叶斯分类器训练函数   ##########
def trainNB0_Sparse(row,col,data,trainCategory):
    print "trainNB0_Sparse start:"
    lenTrainCategory = len(trainCategory)
    print "len(trainCategory): ",lenTrainCategory
    numTrainDocs = len(set(row))
    print "len(row):",len(row)
    print "len(set(row)) = numTrainDocs:",numTrainDocs

    if(numTrainDocs != lenTrainCategory):
        print "input data and class are not equal,len(input data):%d,class:%d" %(numTrainDocs,lenTrainCategory)
    else:print "input data and class are equal"


    numWords = max(col)+1  #相当于trainMat[0]的长度
    print numWords
    print 'len(trainMatrix[0]):',numWords
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords);p1Num=ones(numWords)
    p0Num_20000 = zeros(numWords);p1Num_20000 = zeros(numWords)

    p0Denom = 2.0;p1Denom =2.0
    #p1Temp = zeros(numWords)
    import traceback

    for i in  range (numTrainDocs):
        if trainCategory[i] == 1:
            p1Temp = zeros(numWords)

            rowIndex = row.index(i)
            if(i < numTrainDocs-1):
                rowIndexPlus = row.index(i+1)
            else: rowIndexPlus = len(row)

            for j in range(rowIndex,rowIndexPlus):

                p1Temp[col[j]] += 1

            p1Num_20000 += p1Temp
            if(i > 120000 or i % 10000 == 0):
                p1Num += p1Num_20000
                p1Num_20000 = zeros(numWords)

            if(i > 120000 or i % 100 == 0): print i, "p1Num:",p1Num
            p1Denom += sum(p1Temp)
        else:

            p0Temp = zeros(numWords)
            try:
                rowIndex = row.index(i)
            except ValueError:
                traceback.print_exc()
            while(row[rowIndex]==i):
                p0Temp[col[rowIndex]] += 1
                rowIndex += 1
            p0Num_20000 += p0Temp
            if(i > 60000 or i % 10000 == 0):
                p0Num += p0Num_20000
                p0Num_20000 = zeros(numWords)
            if(i % 1000 == 0):print i, " p0Num ",p0Num
            p0Denom += sum(p0Temp)

    # p1Vect = p1Num / p1Denom      这是算p（wi|c1）的矩阵，已知在class=1的条件下，w0的概率就是w0/(class=1时总的词条数)
    #p0Vect = p0NUm / p1Denom       p（wi|c1）的矩阵

    save("dataset/final/p0Num2.npy",p0Num)
    save("dataset/final/p1Num2.npy",p1Num)
    save("dataset/final/p0Denom2.npy",p0Denom)
    save("dataset/final/p1Denom2.npy",p1Denom)
    print "len(p1Num):",len(p1Num)
    print "len(p0Num):",len(p0Num)
    p1Vect = log(p1Num/p1Denom)   #change to log()  p(w0|1)*p(w1|1)*p(w2|1)..很多极小的数相乘，最后四舍五入会得到0，造成下溢出，所以取对数
    p0Vect = log(p0Num/p0Denom)  #change to log()
    save("dataset/final/p0Vect2.npy",p0Vect)
    save("dataset/final/p1Vect2.npy",p1Vect)
    print "pAbusive",pAbusive
    print type(p0Vect)
    print type(p0Vect)
    fp = open("dataset/final/pAbusive2.txt",'w')
    fp.write(str(pAbusive))

    return p0Vect,p1Vect,pAbusive

##########   贝叶斯分类器  #########

"""
# vec2Classify:要分类的向量
# p0Vec:p(wi|c0)
# p1Vec:p(wi|c1)
# pClass1:p(c1)
# p(ci|w)正比于p(w|ci)*p(ci),因为分母相同，都是p(w)
"""
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)   #对应元素相乘，然后将所有词的对应值相加，然后将该值加到类别的对数概率上  logA+logB+logC=log(A*B*C)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 -pClass1)
    if p1 >p0 :
        return  1  #"差评"
    else:
        return  0  #"好评"


###    装载数据集    ###
#返回  listPosts      数据集列表
#      listClasses    分类列表
def loadDataSet(positive,negative):
    listPosts,listClasses=testTextParse(positive,0)  #0:positive
    print 'positive len(listPosts):',len(listPosts)
    listPosts1,listClasses1=testTextParse(negative,1)  #1:negative
    print ' negative len(listPosts1):',len(listPosts1)
    listPosts += listPosts1
    listClasses =list(listClasses)+list(listClasses1)
    listClasses = array(listClasses)
    print "positive+negative:len(listPosts):",len(listPosts)  #4000
    print "positive+negative:len(listClasses):",len(listClasses)
    return listPosts,listClasses

#####  词集模型  #####
def setOfWords2Vec_Sparse(myVocabList,listPosts):
    index = 0;row=[];col=[];data=[]
    for postinDoc in listPosts:  #在4000条记录中
        #returnVec=[0]*len(vocabList)
        for word in postinDoc:
            if word in myVocabList:
                row.append(index)
                col.append(myVocabList.index(word))
                data.append(1)
                #returnVec[vocabList.index(word)] = 1
            else: print "the word: %s is not in my Vocabulary!"% word
        index += 1
    return row,col,data


##### 稀疏矩阵测试  ######
def testingNBChinese_Sparse(fileName):
    fp = open("testlog.txt","w+")
    logg = "\n" + str(time.clock())
    fp.write(logg)
    # listPosts,listClasses = loadDataSet("dataset/phoneGood.txt","dataset/phoneBad.txt")
    listPosts, listClasses = loadDataSet("dataset/NLPgoodComment_division.txt","dataset/NLP6wbadresult_division.txt")
    print "listClasses:",listClasses
    myVocabList=createVocabList(listPosts)  #得到词典
    save("dataset/final/myVocabList.npy",myVocabList)

    print 'len(myVocabList):',len(myVocabList)  #0:8330  1:15844-8330
    index = 0;row=[];col=[];data=[]
    for postinDoc in listPosts:  #在4000条记录中
        #returnVec=[0]*len(vocabList)
        for word in postinDoc:
            if word in myVocabList:
                row.append(index)
                col.append(myVocabList.index(word))
                data.append(1)

        index += 1
    save("dataset/final/row.npy",row)
    save("dataset/final/col.npy",col)
    save("dataset/final/data.npy",data)

    row=load("dataset/final/row.npy")
    col=load("dataset/final/col.npy")
    data=load("dataset/final/data.npy")
    print row
    print col
    print data
    print len(row),len(col),len(data)
    comment0 = list(row).index(1)
    print comment0
    for i in range(comment0):
            print i,col[i],myVocabList[col[i]].decode('utf-8','ignore')

    p0V,p1V,pSpam = trainNB0_Sparse(list(row),list(col),list(data),array(listClasses))


    logg =  "trainMat building end. "+str(time.clock())+"\n"
    fp.write(logg)
    print logg

    logg = "test start:"+str(time.clock())+"\n"
    fp.write(logg)
    testEntry=testText(fileName)
    testResult=[]
    labelCounts = {}#字典
    leng = len(testEntry)
    for i in range(leng):
        testDoc=setOfWords2Vec(myVocabList,testEntry[i])
        result = classifyNB(testDoc,p0V,p1V,pSpam)
        testResult.append(result)
        if result not in labelCounts.keys():  #.keys():返回一个字典所有的键
            labelCounts[result] = 0
        labelCounts[result] += 1
        print 'testEntry classified as: ',result
    print testResult
    print labelCounts
    logg = "test end." + str(time.clock())
    fp.write(logg)
    print "OK"


def classify_Chinese_sparse(fileName):
    myVocabList = load("dataset/final/myVocabList.npy")
    row=load("dataset/final/row.npy")
    col=load("dataset/final/col.npy")
    data=load("dataset/final/data.npy")
    listPosts, listClasses = loadDataSet("dataset/NLPgoodComment_division.txt","dataset/NLP6wbadresult_division.txt")
    print "listClasses:",listClasses

    listClasses = load("dataset/final/listClasses.npy")
    print "listClasses:",listClasses

    p0V = load("dataset/final/p0Vect.npy")
    p1V = load("dataset/final/p1Vect.npy")
    pSpam = open("dataset/final/pAbusive.txt",'r').readline()
    pSpam = float(pSpam)


    print pSpam
    myVocabList = list(myVocabList)

    logg = "test start:"+str(time.clock())+"\n"
    print logg
    if ".txt" in fileName:
        testEntry=testText(fileName)
    else:
        testEntry = fileName
    testResult=[]
    labelCounts = {}#字典
    leng = len(testEntry)
    for i in range(leng):
        testDoc=setOfWords2Vec(myVocabList,testEntry[i])
        result = classifyNB(testDoc,p0V,p1V,pSpam)
        testResult.append(result)
        if result not in labelCounts.keys():  #.keys():返回一个字典所有的键
            labelCounts[result] = 0
        labelCounts[result] += 1
        print 'testEntry classified as: ',result
    print testResult
    print labelCounts
    logg = "test end." + str(time.clock())
    print logg
    print "OK"
    return testResult


########       Sparse  交叉验证     #########
def crossVarify_Sparse(positive,negative,num=2,times=200.0):
    """
    #input  正面情感：positive,
    #       负面情感：negative,
    #       验证次数：默认2次
    #       测试数据：默认10%,可以输入数据量（如300次或400次）或者输入times<1,如0.2，则取20%的数据进行测试
    #output：errorRate[[TP,FN,FP,TN,float(TP)/(TP+FP),float(TP)/(TP+FN),float(errorCount)/len(testSet)],...]
            :param positive:
            :param negative:
            :param num:
            :param times:
            :return:
    """
    errorRate=[]
    for k in range(num):    #交叉验证取平均
        #print "第",k,"次测试："
        testSet=[]
        listPosts,listClasses = loadDataSet(positive,negative)

        lengListPosts = len(listPosts)
        print lengListPosts
        trainSet=range(lengListPosts)   #得到一个listPosts长度的列表，便于后面的计算（del操作，del不能删除numpy.array的元素）
        if(times == 200):
            times = lengListPosts/10
        if(times < 1):
            times = lengListPosts*times
        for i in range(int(times)):
            randIndex = int(random.uniform(0,len(trainSet)))  #返回 0 - len(trainSet)之间的一个随机数
            # if randIndex not in randSet:
            #     randSet.append(randIndex)
            testSet.append(trainSet[randIndex])
            del(trainSet[randIndex])  #把测试数据从训练数据里移除

        print "len(testSet):",len(testSet)
        print "testSet :",testSet
        trainPosts=[];trainClasses=[]#trianPosts:去掉训练数据后的词条矩阵，用于生成词列表
        #row=[];col=[];data=[] # 稀疏矩阵表达
        ### for循环：得到训练数据集  ###
        ###  先得到训练的词列表  ###
        for docIndex in trainSet:
            trainPosts.append(listPosts[docIndex])
            trainClasses.append(listClasses[docIndex])
        myVocabList = createVocabList(trainPosts)

        row,col,data=(setOfWords2Vec_Sparse(myVocabList,trainPosts))
        print len(set(row))
        #return
        for i in range(row.index(0),row.index(1)):
            print myVocabList[col[i]],

        #print "trainClasses:",trainClasses
        p0V,p1V,pSpam = trainNB0_Sparse(row,col,data,array(trainClasses))    #不用多进程，就用trainMat,
        #p0V,p1V,pSpam = trainNB0(array(trainMat[0]),array(trainClasses))  #用多进程，就用trainMat[0]
        errorCount = 0
        TP = 0;FN = 0;FP = 0;TN = 0;precision = 0;recall = 0
        print "errorCount=0"
        for docIndex in testSet:
            truthClassify = listClasses[docIndex]
            wordVector = setOfWords2Vec(myVocabList,listPosts[docIndex])
            testClassify = classifyNB(array(wordVector),p0V,p1V,pSpam)
            print "第%d次交叉测试 第%d个测试分类：%d,实际分类：%d" %(k+1, docIndex,testClassify,truthClassify)
            if truthClassify == testClassify:
                if truthClassify == 0:
                    TP +=1
                elif truthClassify == 1:
                    TN +=1

            else:  #分类错误
                if truthClassify == 0:
                    FN += 1
                elif truthClassify == 1:
                    FP += 1
                errorCount += 1
            #         |　　0　　｜　　1　　｜
            # 0(正例) |    TP   |   FN    |
            # 1(反例) |    FP   |   TN    |
            # if testClassify != truthClassify:
            #     errorCount += 1


        print "errorCount:",errorCount
        errorRate.append([TP,FN,FP,TN,float(TP)/(TP+FP),float(TP)/(TP+FN),float(errorCount)/len(testSet)])
    print errorRate
    return errorRate

def final_verify(fileName,myVocabList,p0V,p1V,pSpam):
    if ".txt" in fileName:
        testEntry=testText(fileName)
    else:
        testEntry = fileName
    testResult=[]
    labelCounts = {}#字典
    leng = len(testEntry)
    for i in range(leng):
        testDoc=setOfWords2Vec(myVocabList,testEntry[i])
        result = classifyNB(testDoc,p0V,p1V,pSpam)
        testResult.append(result)
        if result not in labelCounts.keys():  #.keys():返回一个字典所有的键
            labelCounts[result] = 0
        labelCounts[result] += 1
        print 'testEntry classified as: ',result
    print testResult
    print labelCounts
    logg = "test end." + str(time.clock()) +"\n"
    print logg
    print "OK"
    return testResult


def final():
    """
    get data (userid,text)from mysql,
    Parsing the text,
    then put into naive bayes,
    and compare with the mysql datasheet qy_phone_brand
    if text has phone brand in qy_phone_brand, record it,and classify
    finally, save the result into datasheet qy_phone_emotion
    """
    conn = pymysql.connect(host='10.10.21.21', port=3306, user='root', passwd='123456', db='3s_project',charset='utf8')
    cur = conn.cursor()
    user=[]
    sentence = "select * from user_timeline"
    print sentence
    cur.execute(sentence)
    a = cur.fetchall()
    for i in a:   #[1]:userid  [4]:text
        seg = []
        seg_list = jieba.cut(i[4])

        for j in seg_list:
            seg.append(j)
        # print(' '.join(seg_list))
        b = str(i[2])[0:-11] + str(i[2])[-5:]
        bb = (time.strptime(b,'%a %b %d %H:%M:%S %Y'))
        bbb = time.strftime('%Y-%m-%d %H:%M:%S',bb)
        bbbb = time.mktime(bb)
        print bbb
        print bbbb
        user.append([i[1],seg,i[4],bbb,bbbb])  #parsing
        print i[1],i[4]
    print user

    # compare with qy_phone_brand
    brand = []
    sentence = "select * from qy_phone_brand"
    cur.execute(sentence)
    b = cur.fetchall()
    for i in b:
        brand.append([i[0],i[1].lower()])
    print brand

    p0V = load("dataset/final/p0Vect.npy")
    p1V = load("dataset/final/p1Vect.npy")
    pSpam = open("dataset/final/pAbusive.txt",'r').readline()
    pSpam = float(pSpam)
    myVocabList = load("dataset/final/myVocabList.npy")
    time_flag = 0
    for k in user:
        for i in k[1]: # k[1]:seg
            for j in brand:
                if i.lower() == j[1]:  # j[1] : phone_brand
                    print i
                    # test_result = classify_Chinese_sparse([k[1]])
                    test_result = final_verify([k[1]],list(myVocabList),p0V,p1V,pSpam)
                    test_result = test_result[0]
                    for m in k[1]:
                        if m in ["不好","垃圾","烂"]:
                            test_result = 1
                    if test_result == 0 :
                        test_result = 1  # good emotion
                    else:
                        test_result = -1 # bad emotion
                    for content in k[1]:
                        print content,
                    # check whether the publish_time is newest
                    sentence = ("select unix_timestamp(publish_time) from qy_phone_emotion where userid = " + str(k[0])
                                + " and phone_brand_id = " + str(j[0]) )
                    print sentence
                    cur.execute(sentence)

                    for publish in cur.fetchall():
                        publish_time = publish[0]
                        print "publish_time:",publish[0]#,len(publish_time[0])

                    if publish_time >= k[4]:
                        print "publish_time >= k[4]"
                        break
                    else:
                    #update
                        sentence = ("update qy_phone_emotion set text = '" +
                                    str(k[2]) +"',emotion = " + str(test_result) +
                                    ",recommend = 0"  +
                                    ", publish_time = '" + str(k[3]) +
                                    "' where userid = " + str(k[0])
                                     + " and phone_brand_id = " + str(j[0]))
                        print sentence
                        cur.execute(sentence)
                    sentence = ("insert into qy_phone_emotion values("
                                + str(k[0]) + ",'" + str(k[2]) + "'," + str(j[0])
                                + "," + str(test_result) +"," + str(0)
                                + "," + str(k[3]) + ")" )#0：field recom = 0
                   # cur.execute(sentence)
                    print sentence
                    break


    conn.commit()
    cur.close()
    conn.close()


if __name__ == '__main__':
    #multiprocessing.freeze_support()
    import time
    t0 = time.clock()
    final()
    # testingNBChinese_Sparse("dataset/amazon.txt")
    # classify_Chinese_sparse("dataset/amazonGood.txt")
    #crossVarify_Sparse("dataset/phoneGood.txt","dataset/phoneBad.txt",2,2000)
    # result=crossVarify_Sparse("dataset/NLPgoodComment_division.txt","dataset/NLPbadComment_division.txt",1,0.1)
    # print result
    print "time:",time.clock() - t0,"seconds"

    #[[5334, 303, 45, 219, 0.9916341327384273, 0.9462480042575838, 0.058973055414336555],
    # [5337, 296, 49, 219, 0.9909023393984404, 0.9474525119829575, 0.058464667005592275],
    # [5339, 277, 46, 239, 0.9914577530176416, 0.9506766381766382, 0.05473648534146755],
    # [5349, 308, 41, 203, 0.9923933209647495, 0.9455541806611278, 0.05914251821725131],
    # [5368, 258, 46, 229, 0.9915035094200222, 0.9541414859580519, 0.051516692086087106],
    # [5359, 291, 47, 204, 0.9913059563448021, 0.948495575221239, 0.05727842738518895],
    # [5387, 278, 43, 193, 0.9920810313075507, 0.9509267431597529, 0.05439755973563803],
    # [5376, 248, 54, 223, 0.9900552486187846, 0.9559032716927454, 0.05117776648025758],
    # [5370, 289, 45, 197, 0.9916897506925207, 0.9489309065205866, 0.05660057617352991],
    # [5364, 293, 49, 195, 0.99094771845557, 0.9482057627717871, 0.057956278596847995]]

   # testingNBChinese_Sparse("dataset/amazonGoodAndBad.txt")
    #testingNBChinese("dataset/amazonGoodAndBad.txt")

