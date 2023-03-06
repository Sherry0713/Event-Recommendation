
import random
import time
import csv
from math import sqrt

import numpy as np
from numpy import savetxt
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

def memoize(function):
  memo = {}
  def wrapper(*args):
    
    if args in memo:
      return memo[args]
    else:
      rv = function(*args)
      memo[args] = rv
      return rv
  return wrapper

# get event similarity based on users
def get_event_sim_by_users(id1, id2, exclude):

    # id1: eid
    # id2: eid the same user attended
    # exclude: uid

    # user attended event1
    attend1 = attendance.find({'eid': id1})

    # user attended event2
    attend2 = attendance.find({'eid': id2})
    set1 = set([])
    set2 = set([])
    for a in attend1:
        if 'interested' in a or 'yes' in a:
            set1.add(a['uid'])
    for a in attend2:
        if 'interested' in a or 'yes' in a:
            set2.add(a['uid'])

    # find user interested in both events
    intersection = set1.intersection(set2)

    s = float(len(intersection))
    if exclude in intersection:
        s -= 1

    # s is in range(0,1)
    s /= min(len(set1), len(set2))

    # the greater the s, the more similar these two events

    return s
    
def get_event_sim_by_cluster(user, event):
    f = []
    for key in ['user_taste', 'friends_taste', 'user_hates', \
            'friends_hate', 'user_invited']:
        if key in user:
            s = 0
            taste = user[key]
            if 'cl0' and 'cl1' in event:
                s += taste['cl0'][event['cl0']] * 100
                s += taste['cl1'][event['cl1']] * 400
                f.append(s)
        else:
            f.append(None)
            #f.append(0 if key in ['user_hates', 'friends_hate'] else None)
    return f

@memoize
def get_user_attendance(uid):
    return list(attendance.find({'uid': uid}))

def get_event_attendance(eid):
    return list(attendance.find({'eid': int(eid)}))
    
@memoize
def get_event_similarity_by_user_big(uid, eid):
    yes_sim = []

    # find events user showed attitude before
    attend_list_u = get_user_attendance(uid)

    # find similarity of eid and events the user showed interest before
    for e2 in attend_list_u:
        if ('yes' in e2 or 'maybe' in e2) and e2['eid'] != eid:
            yes_sim.append(get_event_sim_by_users(eid, e2['eid'], uid))

    # similarity of this eid and the events the user showed interest before
    if yes_sim:
        return sum(yes_sim) / len(yes_sim)
    return None

def get_location_distance(l1, l2):
    if not l1 or not l2:
        return None
    
    if isinstance(l1[0], float):
        l1 = [l1]
    if isinstance(l2[0], float):
        l2 = [l2]
    minimum = 100000000
    for a in l1:
        for b in l2:
            d = sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
            if d < minimum:
                minimum = d
    
    return minimum

def get_event_distance(w1, w2):
    # this uses the cosine similarity
    if not w1 or not w2:
        return None
    w1 = np.array(w1)
    w2 = np.array(w2)
    w1 = np.log(w1 + 1)
    w2 = np.log(w2 + 1)
    s = np.sum(w1 * w2) / (np.sqrt(np.sum(w1 * w1)) * np.sqrt(np.sum(w2 * w2)))
    
    return s

def compare_location_string(s1, s2):
    if not s1 or not s2:
        return 0
    s1 = s1.lower()
    s2 = s2.lower()
    jog = ['yogyakarta', 'jogjakarta']
    if s1 in jog and s2 in jog:
        return 1
    return int(s1 == s2)

def process_locations(elocs, ulocs):
    same_country = 0
    same_state = 0
    same_city = 0
    for el in elocs:
        for ul in ulocs:
            same_country = max(same_country, \
                compare_location_string(el.get('country'), ul.get('country')))
            same_state = max(same_state, \
                compare_location_string(el.get('state'), ul.get('state')))
            same_city = max(same_city, \
                compare_location_string(el.get('city'), ul.get('city')))
            if compare_location_string(el.get('country'), ul.get('country')) \
                    and same_city == 0:
                if 'city' in el and 'city' in ul:
                    same_city = 0.5
                elif 'city' in el:
                    same_city = 0.25
    return [same_country + same_state + same_city * 2.0]
    
ATTR = ['yes', 'no', 'maybe', 'invited']
'''
attended_yes = {}
for att in attendance.find():
    if 'yes' in att or 'maybe' in att:
        u = att['uid']
        e = att['eid']
        if u not in attended_yes:
            attended_yes[u] = []
        attended_yes[u].append(e)
'''


def process_events(uid, e_dict):
    # uid and e_dict are from train.csv
    # gets a dict of events to determine a user's interest (e_dict) and a uid
    # get the events related to the user from the db 
    # e_dict has value for invited

    # find events the user attended before
    attend_list_u = get_user_attendance(uid)
    attend_list_ids = [f['eid'] for f in attend_list_u]

    # e_dict.key is eid in train.csv
    # find events info of events that user showed attitude in train.csv
    e_list = list(event_info.find({'eid':{'$in': list(e_dict.keys())}}))


    # find user info of the uid
    user = user_info.find_one({'id': uid})

    # put events user attended info in dict
    attend_dict = {e['eid']: e for e in attend_list_u}

    # find friends of uid
    friends = friends_db.find_one({'uid': int(uid)})

    # put friends id into set
    friend_ids = set(friends['friends'])
    
    features_dict = {}
    for e in e_list:

        # find users who attended the same event
        attend_list_e = get_event_attendance(e['eid'])
        features = [0, 0, 0, 0]
        for att in attend_list_e:
            if 'yes' in att:
                features[0] += 1
            if 'no' in att:
                features[1] += 1
            if 'maybe' in att:
                features[2] += 1
            if 'invited' in att:
                features[3] += 1
        features.extend([
            features[1] * 1.0 / (features[0] + 1),
            features[2] * 1.0 / (features[0] + 1),
            features[3] * 1.0 / (features[0] + 1),
        ])
        
        features2 = [0, 0, 0, 0]

        # find user's friends who also attended the event
        for att in attend_list_e:
            if att['uid'] not in friend_ids:
                continue
            if 'yes' in att:
                features2[0] += 1
            if 'no' in att:
                features2[1] += 1
            if 'maybe' in att:
                features2[2] += 1
            if 'invited' in att:
                features2[3] += 1
        features2.extend([
            features2[1] * 1.0 / (features2[0] + 1),
            features2[2] * 1.0 / (features2[0] + 1),
            features2[3] * 1.0 / (features2[0] + 1),
        ])
        features2.extend([
            features2[0] / (len(friend_ids) + 1.0),
            features2[1] / (len(friend_ids) + 1.0),
            features2[2] / (len(friend_ids) + 1.0),
            features2[3] / (len(friend_ids) + 1.0),
        ])
        features.extend(features2)
        
        # add distance to event
        
        features.extend(
            process_locations(
                e.get('loc', []),
                user.get('newloc', [])
            )
        )
        
        # add age profile
        if user['birth'] != 'None' and isinstance(user['birth'], str) \
                and 'ages' in e:
            d = abs(2013 - int(user['birth']) - e['mean_age'])
            features.append(d + int(random.random() * 6))
        else:
            features.append(None)
        
        # add gender profile
        sexes = {'male':1, 'female':0}

        if user['gender'] != 'None' and isinstance(user['gender'], str) \
                and 'genders' in e:
            features.append((sexes[user['gender']] + 1.0) / \
                    (sexes['male'] + sexes['female'] + 2.0))
        else:
            features.append(None)
        
        # add event similarity by user
        features.append(get_event_similarity_by_user_big(uid, e['eid']))
        
        # add user to event cluster sim
        features.extend(get_event_sim_by_cluster(user, e))

        # add time to event
        features.append(e['start'] - e_dict[e['eid']][1])
        
        # see if event creator is a friend - ar putea cauza scadere de 0.2%
        # this feature is a boolean
        features.append(e['creator'] in friend_ids)
        
        # compare to what the user's friends like
        if 'words' in e:
            if 'prototype' in user:
                features.append(get_event_distance(user['prototype'], e['words']))
            else:
                features.append(None)

            if 'prototype' in user and 'prototype_invite' in user:
                v1 = get_event_distance(user['prototype'], e['words'])
                v2 = get_event_distance(user['prototype_invite'], e['words'])
                if v1 and v2:
                    features.append(v1 - v2)
                else:
                    features.append(None)
            else:
                features.append(None)
            if 'prototype_hate' in user and 'prototype_invite' in user:
                v1 = get_event_distance(user['prototype_hate'], e['words'])
                v2 = get_event_distance(user['prototype_invite'], e['words'])
                if v1 and v2:
                    features.append(v1 - v2)
                else:
                    features.append(None)
            else:
                features.append(None)
            if 'prototype_hate' in user and 'prototype' in user:
                v1 = get_event_distance(user['prototype_hate'], e['words'])
                v2 = get_event_distance(user['prototype'], e['words'])
                if v1 and v2:
                    features.append(v1 - v2)
                else:
                    features.append(None)
            else:
                features.append(None)
        
        # add invited flag
        features.append(e_dict[e['eid']][0])
            
        features_dict[e['eid']] = features


    return features_dict
    
def write_submission(submission_name, user_events_dict):
    users = sorted(user_events_dict)
    events = [' '.join([str(s) for s in user_events_dict[u]]) for u in users]

    submission = pd.DataFrame({"User": users, "Events": events})
    submission[["User", "Events"]].to_csv(submission_name, index=False)
    
'''
def get_train_data():
    train = pd.read_csv( "/home/ruifan/Downloads/event-recommendation-engine-challenge/train.csv")
    train_dict = {}
    for record in train.iterrows():
        record = record[1]
        uid = record['user']
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append({
            'eid': record['event'],
            'invited': record['invited'],
            'interested': record['interested'],
            'not_interested': record['not_interested']
        })
        
    X = []
    Y1 = []
    Y2 = []    
    count = train['user'].count()
    for uid, events in train_dict.iteritems():
        e_dict = {e['eid']: e['invited'] for e in events}
        features_dict = process_events(uid, e_dict)
        print len(X) * 100.0 / count
        for e in events:
            eid = e['eid']
            X.append(features_dict[eid])
            Y1.append(e['interested'])
            Y2.append(e['not_interested'])
    
    return (X, Y1, Y2)
'''

def get_crossval_data():
    train = pd.read_csv( "/home/ruifan/Downloads/event-recommendation-engine-challenge/train.csv")
    train_dict = {}
    duplicates = set([])
    for record in train.iterrows():
        record = record[1]
        uid = record['user']
        
        # check for duplicates
        key = (uid, record['event'])
        if key in duplicates:
            continue
        duplicates.add(key)
        
        if uid not in train_dict:
            train_dict[uid] = []

        # train_dict: key is uid, value is the events user attended
        train_dict[uid].append({
            'eid': record['event'],
            'invited': record['invited'],
            'interested': record['interested'],
            'not_interested': record['not_interested'],
            'timestamp': time.mktime(parse(record['timestamp']).timetuple()),
        })
      
    splits = []
    irange = [0, len(train_dict) // 2, len(train_dict)]
    for i in range(2):
        X = []
        Y1 = []
        Y2 = []
        results = {}
        keys = []
        count = train['user'].count()
        for uid, events in list(train_dict.items())[irange[i] : irange[i + 1]]:
            e_dict = {e['eid']: (e['invited'], e['timestamp']) for e in events}
            features_dict = process_events(uid, e_dict)
            results[uid] = []
            for e in events:
                eid = e['eid']
                if eid in features_dict.keys():
                    X.append(features_dict[eid])
                    Y1.append(e['interested'])
                    Y2.append(e['not_interested'])
                    keys.append((uid, e['eid']))
                    if e['interested']:
                        results[uid].append(e['eid'])
        
        splits.append((X, Y1, Y2, results, keys))
        
    return splits
    
def get_test_data():
    solutions_dict = get_test_solutions()
    test = pd.read_csv("/home/ruifan/Downloads/event-recommendation-engine-challenge/test.csv")
    test_dict = {}
    for record in test.iterrows():
        record = record[1]
        uid = record['user']
        if uid not in solutions_dict:
            continue
        if uid not in test_dict:
            test_dict[uid] = []
        test_dict[uid].append({
            'eid': record['event'],
            'invited': record['invited'],
            'timestamp': time.mktime(parse(record['timestamp']).timetuple()),
        })
        

    test_data = {}
    for uid, events in test_dict.items():
        e_dict = {e['eid']: (e['invited'], e['timestamp']) for e in events}
        features_dict = process_events(uid, e_dict)
        X = []
        E = []
        count = 0
        for e in events:
            eid = e['eid']
            if eid in features_dict:
                X.append(features_dict[eid])
                E.append(eid)
        test_data[uid] = { 
            'X': X,
            'events': E
        }
        
    return test_data

def get_final_data():
    final_file = pd.read_csv("/home/ruifan/Downloads/event-recommendation-engine-challenge/event_popularity_benchmark_private_test_only.csv")
    final_dict = {}
    final_file['Events'] = final_file['Events'].str.replace('L', '')
    for row in final_file.iterrows():
        row = row[1]
        uid = int(row['User'])
        eid = eval(row['Events'])
        eid = list(eid)
        if uid in final_dict:
            raise Exception
        final_dict[uid] = eid
    return final_dict

def run_model(m1, test_data, is_final=True):
    final_dict = get_final_data()
    results = {}
    for uid, record in test_data.items():
        if is_final and uid not in final_dict:
            continue
        X = record['X']
        events = record['events']
        if X:
            Y1 = m1.test(X)
            final = Y1
            sorted_events = []
            for i, e in enumerate(events):
                score = final[i]
                sorted_events.append((e, score))
            sorted_events.sort(key=lambda x: -x[1])
            sorted_events = [e[0] for e in sorted_events]
            results[uid] = sorted_events
    
    return results
     
def run_crossval():
    splits = get_crossval_data()

    del splits[0][0][1713]
    del splits[0][1][1713]
    del splits[0][2][1713]
    del splits[0][4][1713]
    del splits[1][0][1100]
    del splits[1][1][1100]
    del splits[1][2][1100]
    del splits[1][4][1100]


    results = []
    for i in range(2):
        s = splits[i]
        other_s = splits[1 - i]
        
        w = [True] * len(s[0][0])

        remove_features_lr = [19,20,21,22,23,24,25,26,29,30,31,32]
        

        for i in remove_features_lr:
            w[i] = False

        m1 = Model(compress=w, has_none=None)
        m1.fit(s[0], s[1])


        X = other_s[0]
        predictions = m1.test(X)

        keys = other_s[4]
        pred_dict = {}
        for j in range(len(keys)):
            uid, eid = keys[j]
            if uid not in pred_dict:
                pred_dict[uid] = []
            pred_dict[uid].append((eid, predictions[j]))
            
        for uid, l in pred_dict.items():
            l.sort(key=lambda x: -x[1])
            l = [e[0] for e in l]
            results.append(apk(other_s[3][uid], l))
        
    print(sum(results) / len(results))

def get_test_solutions():
    # read solutions
    solutions_file = pd.read_csv("/home/ruifan/Downloads/event-recommendation-engine-challenge/public_leaderboard_solution.csv")
    solutions_dict = {}
    for row in solutions_file.iterrows():
        row = row[1]
        uid = int(row['User'])
        eid = int(row['Events'])
        solutions_dict[uid] = [eid]
    return solutions_dict

def evaluate_test_results(my_results):
    solutions_dict = get_test_solutions()
    scores = []
    for uid, l in my_results.items():
        score = apk(solutions_dict[uid], l)
        scores.append(score)
    return sum(scores) / len(scores)

def run_full():
    splits = get_crossval_data()

    X = splits[0][0] + splits[1][0]
    Y1 = splits[0][1] + splits[1][1]
    Y2 = splits[0][2] + splits[1][2]

    del_index = [1713,5761]
    del X[1713]
    del X[5761]

    del Y1[1713]
    del Y1[5761]



    test_data = get_test_data()

    t = set()



    remove_features_rfc = [19,20]
    remove_features_lr = [19,20,21,22,23,24,25,26,29,30,31,32]
    
    not_useful_rfc = [8,11,22,24,28,33,30,31,32]
    remove_features_rfc.extend(not_useful_rfc)
    not_useful_lr = [3,4,9,11,14,15,16,17,27,28,30,31,32]
    remove_features_lr.extend(not_useful_lr)

    remove_features = [3, 4, 8, 9, 11, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
    z = [True] * len(X[0])
    
    for i in remove_features:
        z[i] = False
    
    C = 0.03
    #C = 0.3
    m1 = Model(compress=z, has_none = None, C=C)
    m1.fit(X, Y1)
    final = False
    results = run_model(m1, test_data, is_final=final)
    if not final:
        print(evaluate_test_results(results))
    #write_submission('output.csv', results)

if __name__ == '__main__':
    run_full()
'''
features explanation:
    0 - users attending event
    1 - users not attending event
    2 - users maybe attending event
    3 - users invited to event
    4 - f1 / f0
    5 - f2 / f0
    6 - f3 / f0
    7 - friends attending event
    8 - friends not attending event
    9 - friends maybe attending event
    10 - friends invited to event
    11 - f8 / f7
    12 - f9 / f7
    13 - f10 / f7
    14 - f7 / number of friends
    15 - f8 / number of friends
    16 - f9 / number of friends
    17 - f10 / number of friends
    18 - location
    19 - age
    20 - gender
    21 - similarity between user and event based on attendance
    22 - similarity based on clusters from words - events user attended
    23 - similarity based on clusters from words - events friends attended
    24 - similarity based on clusters from words - events user didn't attend
    25 - similarity based on clusters from words - events friends didn't attend
    26 - similarity based on clusters from words - events user was invited to
    27 - time to event
    28 - event creator is friend
    29 - similarity to word model of what user's friends are attending
    30 - prototype diff 1
    31 - prototype diff 2
    32 - prototype diff 3
    33 - invited
    34 - old location - not used in final model because it uses external data
'''
    
