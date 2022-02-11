import json

from flask import Flask, request, send_from_directory, jsonify
from flask_cors import *

from CALCULATE_MODULE.GNN import repository
from entity.vo.resultvo import Result,SUCCESS,FAIL,REBUT

app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.route('/', methods=['POST', 'GET'])
def index():
    # 1.获取参数
    index = request.args.get('index')

    # 2.执行业务逻辑，返回数据结果
    pass
    # 3.响应结果
    return Result(SUCCESS,"查询成功", index).tostring()

@app.route('/data/graph', methods=['POST', 'GET'])
def get_graph_data():
    # 1.获取参数
    db = request.args.get('db', default="1*1")
    sample = request.args.get('sample', default="8")
    block = request.args.get('block', default="4")
    sub = request.args.get('sub', default="1")
    labels = ["DB%s"%db, "Grain", "Sample%s"%sample, "Block%s"%block, "Sub%s"%sub]
    nodes_data,links_data = repository.read_graph_from_neo4j(labels, "id", "source", "target")

    nodes = []
    edges = []
    for node in nodes_data:
        nodes.append({"id":str(node["id"]),
                      "x":node["properties"]["xAve"],
                      "y":node["properties"]["yAve"],
                      "size":node["properties"]["size"],
                      "olabel":str(node["properties"]["grainID"])
                      })
    for edge in links_data:
        edges.append({"source": str(edge["source"]),"target": str(edge["target"])})
    # 2.执行业务逻辑，返回数据结果
    result = {
        "nodes": nodes,
        "edges":edges
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(port="9007", debug=True) # 运行配置改才生效