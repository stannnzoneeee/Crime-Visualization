#downloader.py
import os
import pandas as pd
import pymongo
from dotenv import load_dotenv
from pymongo import MongoClient
import time

class PeriodicMongoDBDataDownloader:
    def __init__(self, mongo_uri=None, db_name=None, output_dir="./"):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize MongoDB connection details
        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI")
        self.db_name = db_name or os.getenv("MONGO_DB_NAME")
        self.output_dir = output_dir
        
        # Initialize MongoDB client
        self.client = None
        self.db = None
    
    def connect_to_mongodb(self):
        """Establish connection to MongoDB."""
        try:
            if not self.client:
                self.client = pymongo.MongoClient(self.mongo_uri)
                self.db = self.client[self.db_name]
                print(f"✅ Connected to MongoDB database: {self.db_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def download_collection(self, collection_name):
        """Download a single collection and append new data to existing CSV."""
        try:
            # Connect to MongoDB if not already connected
            self.connect_to_mongodb()
            
            collection = self.db[collection_name]
            cursor = collection.find({})
            new_df = pd.DataFrame(list(cursor))
        
            if new_df.empty:
                print(f"No new data in {collection_name}")
                return None
        
            # Handle ObjectId conversion
            if "_id" in new_df.columns:
                new_df["_id"] = new_df["_id"].astype(str)
            
            # Load existing data if the file exists
            csv_path = os.path.join(self.output_dir, f"{collection_name}.csv")
            if os.path.exists(csv_path):
                old_df = pd.read_csv(csv_path)
                # Combine old and new data
                df = pd.concat([old_df, new_df], ignore_index=True)
                # Remove duplicates based on _id
                df = df.drop_duplicates(subset="_id", keep="last")
            else:
                df = new_df
            
            # Save combined data
            df.to_csv(csv_path, index=False)
            print(f"💾 Appended {collection_name} to {csv_path} (Total rows: {len(df)})")
            
            return df
        except Exception as e:
            print(f"❌ Failed to download collection {collection_name}: {e}")
            return None
    
    def download_collections(self, collections):
        """Download multiple collections and return as a dictionary of DataFrames."""
        dataframes = {}
        
        for collection_name in collections:
            df = self.download_collection(collection_name)
            if df is not None:
                dataframes[collection_name] = df
        
        return dataframes
    
    def close_connection(self):
        """Close the MongoDB connection if it exists."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            print("✅ MongoDB connection closed")
    
    def start_single_download(self, collections):
        """Perform a single data download."""
        try:
            print(f"\n🔄 Starting data download at {time.strftime('%Y-%m-%d %H:%M:%S')}")
            dfs = self.download_collections(collections)
        except Exception as e:
            print(f"❌ Error during single download: {e}")
    
    def __del__(self):
        """Close connection when the object is destroyed."""
        self.close_connection()