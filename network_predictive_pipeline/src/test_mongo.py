from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")

db = client["mydb"]

print(
    db["network_data"].count_documents(
        {"category": "Network"}
    )
)