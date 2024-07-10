# main.py
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from animal_embedding import lion_features, tiger_features, elephant_features, giraffe_features, zebra_features

# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define schema for the animal collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="animal_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="features", dtype=DataType.FLOAT_VECTOR, dim=128)
]
schema = CollectionSchema(fields, "Animal Collection")

# Create the animal collection
collection = Collection(name="animal_collection", schema=schema)

# Example animal names
animal_names = ["Lion", "Tiger", "Elephant", "Giraffe", "Zebra"]

# Example feature vectors in a list
features = [
    lion_features,
    tiger_features,
    elephant_features,
    giraffe_features,
    zebra_features
]

# Prepare data for insertion
data = [
    [i for i in range(5)],  # IDs
    animal_names,           # Animal names
    features                # Feature vectors
]

# Insert data into the collection
collection.insert(data)

# Load collection into memory
collection.load()

# Example feature vector for a lion (using the imported embedding)
query_vector = [lion_features]  # Single query vector for the lion

# Define search parameters
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

# Perform a search to find similar animals
results = collection.search(query_vector, "features", search_params, limit=3, output_fields=["id", "animal_name"])

# Filter out the query animal from results
filtered_results = [hit for hit in results[0] if hit.entity.get("animal_name") != "Lion"]

# Print results
print("Query result:")
for hit in filtered_results:
    print(f"ID: {hit.id}, Animal Name: {hit.entity.get('animal_name')}, Distance: {hit.distance}")
