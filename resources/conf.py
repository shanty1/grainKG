import os
from entity.vo.defined_class import Map

path,db = Map(),Map()

path.basedir = os.path.abspath(os.path.dirname(__file__)) + os.path.sep # 项目路径


db=Map({
    "uri": "bolt://192.168.31.26:7688/",
    "username": "neo4j",
    "password": "123456"
})