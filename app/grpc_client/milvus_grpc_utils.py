import sys
import os
import grpc
sys.path.append(os.path.dirname(__file__))
from config import MILVUS_HOST, MILVUS_PORT ,GRPC_HOST, GRPC_PORT
import image_transform_pb2_grpc
import image_transform_pb2
from pymilvus import connections, Collection, list_collections

def show_collections():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collections = list_collections()
    print(f"Collections in Milvus: {collections}")
    return collections

def show_collection_schema(collection_name):
    collection = Collection(collection_name)
    print(f"Schema for '{collection_name}':")
    for field in collection.schema.fields:
        print(f"- {field.name} ({field.dtype})")

def get_vector_from_milvus(collection_name, id):
    collection = Collection(collection_name)
    results = collection.query(expr=f"id == '{id}'", output_fields=["embedding"])
    if results:
        return results[0]["embedding"]
    else:
        raise ValueError("Vector not found")

def show_all_ids(collection_name):
    collection = Collection(collection_name)
    count = collection.num_entities
    print(f"Total entities: {count}")
    if count > 0:
        results = collection.query(expr="", output_fields=["id"], limit=count)
        print("All IDs in collection:", [r["id"] for r in results])
    else:
        print("No entities found in the collection.")




def encode_vector_with_grpc(vector, employee_id,name):

    with open('key/server.crt', 'rb') as f:
        trusted_certs = f.read()

    credentials = grpc.ssl_channel_credentials(root_certificates=trusted_certs)
    grpc_address = f"{GRPC_HOST}:{GRPC_PORT}"
    channel = grpc.secure_channel(grpc_address, credentials)
    stub = image_transform_pb2_grpc.EncodeServiceStub(channel)


    request = image_transform_pb2.VectorRequest(name=name, employee_id=employee_id,vector=vector)

    response = stub.encode_vector(request)
    return response.vector
