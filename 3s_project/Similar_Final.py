# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
@author: Wicle Qian
2015-09-16
calculate the similarity of weibo user
"""
import pymysql
import string
from math import *
from numpy import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def sim_distance_cos_based_phone():
    conn = pymysql.connect(host='10.10.21.21', port=3306, user='root', passwd='123456', db='3s_project',charset='utf8')
    cur = conn.cursor()
    sentence = "select * from qy_phone_emotion where recommend = 0"
    print sentence
    cur.execute(sentence)
    a = cur.fetchall()
    phone = []
    phone_user = []
    phone_brand = []
    for i in a:
        phone.append([i[0],i[2],i[3]])
        if i[0] not in phone_user:
            phone_user.append(i[0])
        if i[2] not in phone_brand:
            phone_brand.append(i[2])
    print phone
    print phone_user
    print phone_brand

    user_brand = []
    list_user_brand = []
    for i in phone_user:
        list_temp = []
        temp = {}
        for j in phone_brand:
            flag = 0
            for k in phone:
                if(i == k[0] and j == k[1]):
                    list_temp.append(k[2])
                    temp[j] = k[2]
                    flag = 1
                    break
            if(flag == 0):
                list_temp.append(0)
                temp[j] = 0

        user_brand.append(temp)
        list_user_brand.append(list_temp)
    print "user_brand:"
    for i in user_brand:
        print i,len(i)
    print "list_user_brand:"
    for i in list_user_brand:
        print i

    import itertools
    leng = [i for i in range(len(user_brand))]
    print leng
    # m = list(itertools.combinations(leng, 2))
    m = list(itertools.permutations(leng,2))
    print (len(m))
    print m
    # 得到cos相似度矩阵[(0,1),(0,2)(0,3)(0,4)(0,5)(0,6)],[(1,0),(1,2)(1,3)(1,4)(1,5)(1,6)],...
    pp = [];p = []
    flag = 0
    for i in m:
        p1 = user_brand[i[0]]
        p2 = user_brand[i[1]]
        p.append(similar(p1,p2))
        flag += 1
        if(flag  == len(user_brand)-1):
            flag = 0
            pp.append(p)
            p = []
    for i in pp:
        print i

    # 得到推荐矩阵
    # predict user's emotion to the phone brand
    # p(1,3) = (cos(u1,u2)*p(2,3) + cos(u1,u3)*p(3,3) + cos(u1,u4)*p(4,4) + ...)/(cos(u1,u2)+cos(u1,u3)+cos(u1,u4)+...)
    import copy
    list_user_brand_copy = copy.deepcopy(list_user_brand)
    for i in range(len(list_user_brand)): # i:第i个用户
        print i
        for j in range(len(list_user_brand[i])): # 第i个用户对第j个手机品牌的喜好
            pij = list_user_brand[i][j]
            # print pij,
            if pij == 0:
                p_sum = sum([k for k in pp[i]])
                #print p_sum
                # a = 0
                # a= [1,2,3] * array([4,5,6]).T
                # print a
                a = []
                for m in range(len(list_user_brand)):
                    if m != i:
                        a.append(list_user_brand[m][j])
                p_cos = sum(a * array(pp[i]).T)
                # print a, p_cos,p_sum
                pij = p_cos/p_sum
                list_user_brand_copy[i][j] = pij
    print "list_user_brand_copy:"
    for i in list_user_brand_copy:
        print i
    # insert or update datasheet qy_phone_emotion
    sentence = ("select * from qy_phone_emotion where recommend = 1")
    cur.execute(sentence)
    a = cur.fetchall()
    recom = []
    if(a):
        for i in a:
            recom.append([i[0],i[2]])
    print recom

    for i in range(len(phone_user)):
        for j in range(len(phone_brand)):
            if(not (user_brand[i].get(phone_brand[j]))): # if the user not have the brand,insert or update
                # update
                if ([phone_user[i],phone_brand[j]] in recom):
                    print "update"
                    sentence = ("update qy_phone_emotion set emotion = " + str(list_user_brand_copy[i][j])
                    + "where userid=" + str(phone_user[i]) + " and phone_brand_id = " + str(phone_brand[j]))
                    print sentence
                    cur.execute(sentence)
                else: #insert
                    print list_user_brand_copy[i][j],
                    sentence = ("insert into qy_phone_emotion(userid, phone_brand_id, emotion, recommend) values ("
                                + str(phone_user[i]) +"," + str(phone_brand[j]) + "," + str(list_user_brand_copy[i][j]) + "," + str(1) + ")")
                    print sentence
                    cur.execute(sentence)
            # elif(user_brand[i].get(phone_brand[j]))
        print "\n"

    conn.commit()
    cur.close()
    conn.close()
    return 0

def similar(p1,p2):
    c = set(p1.keys())&set(p2.keys())
    if  len(c) < 1 :
        return 0
    ss = sum([p1.get(sk)*p2.get(sk) for sk in c])
    sq1 = sqrt(sum([float(pow(sk,2)) for sk in p1.values()]))
    sq2 = sqrt(sum([pow(sk,2) for sk in p2.values()]))
    sq1 = sqrt(sum([float(pow(p1.get(sk),2)) for sk in c]))
    sq2 = sqrt(sum([float(pow(p2.get(sk),2)) for sk in c]))
    p = float(ss)/(sq1*sq2)
    return p

if __name__ == '__main__':
    sim_distance_cos_based_phone()
