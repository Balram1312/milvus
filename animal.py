import random
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

_HOST = '127.0.0.1'
_PORT = '19530'

_COLLECTION_NAME = 'animal_collection'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'vector_field'

_DIM = 4  # Dimension of the vectors
_INDEX_FILE_SIZE = 32

_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3

animal_vectors = {
    'tiger': [0.2, 0.8, 0.5, 0.1],
    'lion': [0.3, 0.7, 0.6, 0.2],
    'giraffe': [0.5, 0.1, 0.9, 0.3],
    'alligator': [0.6, 0.2, 0.7, 0.4],
    'deer': [0.1, 0.9, 0.4, 0.5]
}

def create_connection():
    print("\nCreating connection...")
    connections.connect(host=_HOST, port=_PORT)
    print("List of connections:")
    print(connections.list_connections())

def create_collection(name, id_field, vector_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    collection = Collection(name=name, schema=schema)
    print(f"\nCollection created: {name}")
    return collection

def has_collection(name):
    return utility.has_collection(name)

def drop_collection(name):
    collection = Collection(name)
    collection.drop()
    print(f"\nDropped collection: {name}")

def list_collections():
    print("\nList of collections:")
    print(utility.list_collections())

def insert_animals(collection):
    data = [
        [i for i in range(len(animal_vectors))],
        [vector for vector in animal_vectors.values()],
    ]
    collection.insert(data)
    return data[1]

def get_entity_num(collection):
    print("\nNumber of entities in collection:")
    print(collection.num_entities)

def create_index(collection, field_name):
    index_param = {
        "index_type": _INDEX_TYPE,
        "params": {"nlist": _NLIST},
        "metric_type": _METRIC_TYPE
    }
    collection.create_index(field_name, index_param)
    print("\nCreated index:")
    print(collection.index())

def drop_index(collection):
    collection.drop_index()
    print("\nDropped index")

def load_collection(collection):
    collection.load()

def release_collection(collection):
    collection.release()

def search_similar_animal(collection, vector_field, id_field, animal_name):
    search_vector = animal_vectors[animal_name]
    search_param = {
        "data": [search_vector],
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "limit": _TOPK,
        "expr": "id_field >= 0"
    }
    results = collection.search(**search_param)
    print(f"\nSimilar animals to '{animal_name}':")
    for result in results[0]:
        similar_animal_id = result.id
        similar_animal_name = list(animal_vectors.keys())[similar_animal_id]
        print(f"- {similar_animal_name}")

def set_properties(collection):
    collection.set_properties(properties={"collection.ttl.seconds": 1800})

def main():
    create_connection()

    if has_collection(_COLLECTION_NAME):
        drop_collection(_COLLECTION_NAME)

    collection = create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)

    list_collections()

    insert_animals(collection)

    set_properties(collection)

    get_entity_num(collection)

    create_index(collection, _VECTOR_FIELD_NAME)

    load_collection(collection)

    search_similar_animal(collection, _VECTOR_FIELD_NAME, _ID_FIELD_NAME, 'tiger')

    release_collection(collection)

    drop_index(collection)

    drop_collection(_COLLECTION_NAME)

if __name__ == '__main__':
    main()
