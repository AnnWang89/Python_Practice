# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 14:17:39 2021

@author: 安ㄢ
"""

import Algorithmia

input = {
  "image": "data://Beluga/mlbook/black1.jpg"
}
try:
    client = Algorithmia.client('simfSVYs1uHj4lzbfl0I7psZyAP1')
    algo = client.algo('deeplearning/ColorfulImageColorization/1.1.14')
    print(algo.pipe(input).result)
except:
    print("資料圖片檔案讀取錯誤! ")