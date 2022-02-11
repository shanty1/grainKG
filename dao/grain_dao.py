import py2neo
from py2neo import NodeMatcher, RelationshipMatcher
from resources import conf

"""
    注意：数据库中只存原始的晶粒结构数据，不含属性节点
    api
        节点类/关系类：https://py2neo.org/2021.1/data/index.html#node-objects
        节点/关系查询匹配：https://py2neo.org/2021.1/matching.html#node-matching ，返回节点/关系
        创建节点/关系：https://py2neo.org/2021.1/workflow.html#graphservice-objects
"""


def build_neo4j_connection(uri="bolt://localhost:7687", username="neo4j", password="neo4j"):
    graph = py2neo.Graph(uri, auth=(username, password))
    return graph


graph = build_neo4j_connection(conf.db.uri, conf.db.username, conf.db.password)
nodeMatcher = NodeMatcher(graph)
relationMatcher = RelationshipMatcher(graph)


def query_all_labels_in_db():
    """
        查询数据库中的的标签，即所有的标签
        return 如: ["Grain", "No.1", "Sample1"]
    """
    return list(graph.schema.node_labels)


def query_all_labels_of_nodes(graph_database):
    """
    查询节点的标签，即标签的组合（多个标签决定一种节点）
    Param: graph_database  晶粒的切割方法，如1*1，2*2，3*3
    return 如:
    [["DB2*2", "Grain", "Sample1", "Block1", "Sub1"],
     ["DB2*2", "Grain", "Sample1", "Block1", "Sub2"],
     ["DB3*3", "Grain", "Sample1", "Block1", "Sub1"],
     ["Size"],
     ["Ori"]]
    """
    print("查询(%s)的划分图数据库" % graph_database)
    return graph.run("match(n:`%s`) return distinct labels(n) as node_label" % graph_database).to_series().tolist()


def query_all_labels_of_grains(graph_database):
    """
        查询晶粒节点的标签，即标签签的组合（多个标签决定一种节点）
        方法同 $query_all_labels_of_nodes
        Param: graph_database  晶粒的切割方法，如1*1，2*2，3*3
        Return 如:
        [["DB2*2", "Grain", "Sample1", "Block1", "Sub1"],
        ["DB2*2", "Grain", "Sample1", "Block1", "Sub2"],
        ["DB3*3", "Grain", "Sample1", "Block1", "Sub1"],
    """
    print("查询(%s)的划分图数据库" % graph_database)
    return graph.run(
        "match(n:Grain:`%s`) return distinct labels(n) as node_label" % graph_database).to_series().tolist()


def count_all_labels_of_grains(graph_database):
    """
        查询晶粒节点的标签，即标签签的组合（多个标签决定一种节点）
        方法同 $query_all_labels_of_nodes
        Param: graph_database  晶粒的切割方法，如1*1，2*2，3*3
        Return 如:
        [["DB2*2", "Grain", "Sample1", "Block1", "Sub1"],
        ["DB2*2", "Grain", "Sample1", "Block1", "Sub2"],
        ["DB3*3", "Grain", "Sample1", "Block1", "Sub1"],
    """
    print("查询(%s)的划分图数据库" % graph_database)
    return graph.run(
        "match(n:Grain:`%s`) return count(distinct labels(n))" % graph_database).evaluate()


def count_r_with_different_nodes_label():
    """
    查询数据库中不同类别的节点的边数量
    （可以用于验证晶粒图是否正确，不同切割方法、样本、块的晶粒之间应该不可能存在相连，相同以上条件下的不同子图可能存在相连，因为子图是根据晶粒的坐标区间划分）
    """
    return graph.run("match (x) -[r]-(y) where labels(x) <> labels(y) return count(r)").evaluate()


def query_r_with_different_nodes_label():
    """
    查询数据库中不同类别的节点的边数量
    （可以用于验证晶粒图是否正确，不同切割方法、样本、块的晶粒之间应该不可能存在相连，相同以上条件下的不同子图可能存在相连，因为子图是根据晶粒的坐标区间划分）
    """
    return graph.run(
        "match (x) -[r]-(y) where labels(x) <> labels(y) return labels(x) as x, labels(y) as y").to_data_frame()


def validate_grain_graph():
    """
        自定义方法： 验证构建的晶粒图谱是否正确，即不同标签的晶粒不可以存在相连，子图标签除外
    """
    df = query_r_with_different_nodes_label()
    num = 0
    for index, row in df.iterrows():
        in_col1_not_in_col2 = list(set(row['x']) - set(row['y']))
        in_col2_not_in_col1 = list(set(row['y']) - set(row['x']))
        if len(in_col1_not_in_col2) > 1 or len(in_col2_not_in_col1) > 1 \
                or "Sub" not in in_col1_not_in_col2[0] or "Sub" not in in_col2_not_in_col1[0]:
            print(in_col1_not_in_col2, in_col2_not_in_col1)
        else:
            num += 1
    print("没有不同标签的晶粒相连，不同子图标签相连的晶粒有%d对！" % df.shape[1])


if __name__ == '__main__':
    graph = build_neo4j_connection(uri=conf.db.uri, username=conf.db.username, password=conf.db.password)
    print(graph)
    validate_grain_graph()
