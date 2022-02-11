import json

from flask import jsonify

SUCCESS = 1
FAIL = 0
REBUT = -1

class Result():
    def __init__(self, no, msg, data):
        self.no = no
        self.msg = msg
        self.data = data

    def no(self,no):
        self.no = no

    def data(self,msg):
        self.msg = msg

    def data(self,data):
        self.data = data

    def tostring(self):
        return json.dumps(self, default=result2dict)

    def tojson(self):
        return jsonify(self)

#定义一个转换函数，将Result类换成json可以接受的类型
def result2dict(std):
    return {
        'no':std.no,
        'msg':std.msg,
        'data':std.data
    }