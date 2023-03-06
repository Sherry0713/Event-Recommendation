from sklearn.cluster import MiniBatchKMeans
import random
import time
from math import sqrt

import numpy as np
import pandas as pd
from dateutil.parser import parse
from eval import apk
from model import Model
from pymongo import MongoClient

'''
    Initialize PyMongo
'''
client = MongoClient('mongodb://localhost:27017')
db = client.recommendation
user_info = db['user_info']
event_info = db['event_info']
attendance = db['event_attendees']
friends_db = db['user_friends']

user_info.create_index('id', unique = True)
event_info.create_index('eid', unique = True)
attendance.create_index('uid')
attendance.create_index('eid')
friends_db.create_index('uid')

cluster_names = [(100, 'cl0'), (400, 'cl1')]
ckeys = [c[1] for c in cluster_names]

event_clusters = {}
for e in event_info.find():
    if 'words' in e:
        event_clusters[e['eid']] = {k: e[k] for k in ckeys}

print('s1')


def insert_user_taste_into_mongodb(attd, utaste, ftaste):
    # load database data, for fast access
    # key is eid, value is a dict with cl0 and cl1 as keys and predicted cluster numbers as values
    # key is uid, value is eid (users showed attitude to)
    user_attd_dict = {}
    for a in attendance.find():
        if attd in a:
            uid = a['uid']
            eid = a['eid']
            if uid not in user_attd_dict:
                user_attd_dict[uid] = []
            user_attd_dict[uid].append(eid)

    # get the clusters for events an user attends:
    for u in user_info.find():
        uid = int(u['id'])
        e_list = user_attd_dict.get(uid,[])

        if not e_list:
            # no events :(
            continue

        taste = {}
        taste.update({
            'cl0': [0] * 100,
            'cl1': [0] * 400
        })

        for e in e_list:
            if e not in event_clusters:
                continue
            c = event_clusters[e]
            for cl in ckeys:
                taste[cl][c[cl]] += 1

        # normalize taste
        for cl in ckeys:
            s = sum(taste[cl]) + 1.0
            taste[cl] = [i / s for i in taste[cl]]

        user_info.update_one({'id': uid},
                             {'$set': {utaste: taste}})

    print('s2')
    # get the clusters for events a user's friends attend

    for u in user_info.find():
        uid = int(u['id'])
        friends = friends_db.find_one({'uid': uid})
        if not friends:
            continue

        friends = friends['friends']
        events = set([])
        for f in friends:
            events.update(user_attd_dict.get(f, []))

        taste = {}
        taste.update({
            'cl0': [0] * 100,
            'cl1': [0] * 400
        })

        sw = 0
        for eid in events:
            if eid not in event_clusters:
                continue
            c = event_clusters[eid]
            for cl in ckeys:
                taste[cl][c[cl]] += 1
            sw = 1

        if sw == 0:
            # no friends or no events :(
            continue

        # normalize taste
        for cl in ckeys:
            s = sum(taste[cl]) * 1.0
            taste[cl] = [i / s for i in taste[cl]]

        # update
        user_info.update_one({'id': uid},
                             {'$set': {ftaste: taste}})
    print('s3')

insert_user_taste_into_mongodb('invited', 'user_invited', 'friends_invited')
insert_user_taste_into_mongodb('yes', 'user_taste', 'friends_taste')
insert_user_taste_into_mongodb('no', 'user_hate', 'friends_hate')
