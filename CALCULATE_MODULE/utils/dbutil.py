import py2neo


def build_neo4j_connection(uri="bolt://localhost:7687", username="neo4j", password="neo4j"):
    graph = py2neo.Graph(uri, auth=(username, password))
    return graph