import pymongo

DB_NAME = "ProjectDB"
USERNAME = "Risith"
PASSWORD = "Risith#1234"
COLLECTION_NAME = "TestData"


client = pymongo.MongoClient(f'mongodb+srv://{USERNAME}:{PASSWORD}@projectcluster.unskd.mongodb.net/{DB_NAME}?retryWrites=true&w=majority')
db = client[DB_NAME]

#db.counters.insert_one({"_id":"recordId","nextId":0})
db.counters.update_one({"_id":"recordId"},{"$inc":{"nextId":1}})
data = db.counters.find_one({"_id":"recordId"})
print(data["nextId"])
#collection = db[COLLECTION_NAME]

#results = collection.find({})

#for data in results:
    #print(data)