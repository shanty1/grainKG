import os
from pathlib import Path

from sklearn.metrics import r2_score

el = r2_score(  [23.50, 35.30, 38.70, 22.00, 21.90]
    , [23.54, 31.90, 36.09, 21.83, 22.12])

uts = r2_score(
 [286.60, 262.10, 279.80, 341.20, 316.10], [286.37, 268.14, 278.11, 328.38, 313.80])

ys=r2_score([124.00, 87.20, 63.10, 186.90, 128.90] , # 预测值
 [121.32, 63.43, 66.84, 183.29, 128.75] ) # 真实值

print(el)