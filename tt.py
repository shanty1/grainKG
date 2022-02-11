import dgl
import torch as th

g1 = dgl.heterograph({
    ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    ('user', 'plays', 'game'): [(0, 0), (1, 0)]
})
g1.nodes['user'].data['h1'] = th.tensor([[0.], [1.], [2.]])
g1.nodes['user'].data['h2'] = th.tensor([[3.], [4.], [5.]])
g1.nodes['game'].data['h1'] = th.tensor([[0.]])
g1.edges['plays'].data['h1'] = th.tensor([[0.], [1.]])

g2 = dgl.heterograph({
    ('user', 'follows', 'user'): [(0, 1), (1, 2)],
    ('user', 'plays', 'game'): [(0, 0), (1, 0)]
})
g2.nodes['user'].data['h1'] = th.tensor([[0.], [1.], [2.]])
g2.nodes['user'].data['h2'] = th.tensor([[3.], [4.], [5.]])
g2.nodes['game'].data['h1'] = th.tensor([[0.]])
g2.edges['plays'].data['h1'] = th.tensor([[0.], [1.]])

# 对于边类型，只允许典型的边缘类型避免歧义。
bg = dgl.batch_hetero([g1, g2] )
list(bg.nodes['user'].data.keys())
list(bg.nodes['game'].data.keys())
list(bg.edges['follows'].data.keys())
list(bg.edges['plays'].data.keys())

print(g2.nodes['game'].data['h1'])
bg.ndata['h1'] ={"game":th.tensor([[1.], [1.]]),
                 "user":th.ones(6,1)}

g3, g4 = dgl.unbatch(bg)
print(g4.nodes['game'].data['h1'])