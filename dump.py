
import csv
from pymongo import MongoClient


client = MongoClient("mongodb://localhost:27017/")
db = client["scriza_db"]
collection = db["user_dump"]


csv_file_path = "readme.csv"

with open(csv_file_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    dump = []
    for row in reader:
        
        dump.append({
            "ID": row["ID"],
            "Name": row["Name"],
            "Address": row["Address"],
            "City": row["City"],
            "State": row["State"],
            "Zip": row["Zip"],
            "Mobile": row["Mobile"]
        })

if dump:
    collection.insert_many(dump)
    print(f" Insertd {len(dump)} records into MongoDB.")
else:
    print("No data found in CSV.")