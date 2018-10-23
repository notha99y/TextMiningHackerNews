import os
import glob
import time
import json

from pymongo import MongoClient

if __name__ == '__main__':
    data_path = os.path.join(os.getcwd(), 'data', 'raw')
    raw_jsons = glob.glob(os.path.join(data_path, '*.json'))

    print("Please select a raw data.json to put into mongodb")
    for i, file in enumerate(raw_jsons):
        print('[{}]: {}'.format(i, file))
    selected = int(
        input("Input number from 0 - {}: ".format(len(raw_jsons) - 1)))
    print("selected [{}]: {}".format(selected, raw_jsons[selected]))
    # time.sleep(5)
    with open(raw_jsons[selected]) as fp:
        selected_json = json.load(fp)

    client = MongoClient()
    db = client.HackerNews
    comments = db.before_2015_08_08
    for i in selected_json:
        comments.insert(i)
