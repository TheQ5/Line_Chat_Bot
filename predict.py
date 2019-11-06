#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding="UTF-8" -*- 

from IPython.display import clear_output
from pytorch_transformers import BertTokenizer
import torch
import pandas as pd

'''

定義一個可以input文字和可以oupput預測判斷的class。


'''
class sentimentModel():
    
    def __init__(self, text=None, tokenizerName="bert-base-chinese"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.predict_map = {0:"High Positive", 1: "Clam Positive", 2:"High Negative", 3:"clam Negative"}
        self.modelPath = "bert_sentiment_wordmax_128_loss_0.033_lr_2e-05.pkl"
        self.tokenizer = BertTokenizer.from_pretrained(tokenizerName)
        self.text = text
        
    r'''
    設計把一句話轉成bert相容的形式。
    
    input : model的路徑
    output : model you want
    '''
    def loadModel(self, modelPath):

        # load model and set to cuda if they are here
        model = torch.load(modelPath, map_location=self.device)
        model = model.to(self.device)

        print(f"Device:{self.device}")
        return model
    

    r'''
    設計把一句話轉成bert相容的形式。

    input : 一段話，最大長度不超過80個字
    output : 
            token_tensor : 把文字轉成電腦可以理解的方式
            segment_tensor : 皆設為1
            mask_tensor : self attention 關注的地方
    '''
    def convert_text_to_bertEat(self, text):

        if type(text) != str:
            raise TypeError("Input must be str.")
        elif len(text) > 128:
            raise ValueError("the len(s) must less than 128.")

        # 取得3個tensor
        token = self.tokenizer.tokenize(text)
        word_cls = ["[CLS]"]
        word_cls += token
        word_cls_len = len(word_cls)
        ids = self.tokenizer.convert_tokens_to_ids(word_cls)
        token_tensor = torch.tensor(ids)

        segment_tensor = torch.tensor([1] * word_cls_len)
        
        mask_tensor = torch.zeros(token_tensor.shape)
        mask_tensor = mask_tensor.masked_fill(token_tensor !=0, 1)

        # covert tensor, 1D to 2D
        token_tensor = token_tensor.view(-1,token_tensor.size(0))
        segment_tensor = segment_tensor.view(-1,segment_tensor.size(0))
        mask_tensor = mask_tensor.view(-1,mask_tensor.size(0))

        return token_tensor, segment_tensor, mask_tensor


    r'''
    設計一個可以預測的函式，
    input : 能被bert吃的文字
    output : 情感結果
    '''
    def sentimentPredict(self, model, token_tensor, segment_tensor, mask_tensor):
        
        # predict mode
        model.eval()
        with torch.no_grad():
            # move all to cuda
            if torch.cuda.is_available():
                token_tensor = token_tensor.to(self.device)
                segment_tensor = segment_tensor.to(self.device)
                mask_tensor = mask_tensor.to(self.device)

            data = [token_tensor, segment_tensor, mask_tensor]

            outputs = model(*data[:3])
            logits = outputs[0]

            _, pred_num = torch.max(logits.data, 1)

            pred = self.predict_map[pred_num.item()]

        return pred_num.item(), pred
    
    
    r'''
    把結果丟給line chatbot and GUI管理者平台 or 把原文字丟給下一個model
    input : 預測值
    output : 
            1. call百度SDK並把SDK轉換後的文字丟給語意分析的model
            2. call linechatbot message funtion
            3. call 後端管理系統

            待整合，coming soon...
    '''
    def pass_sign_to_lineAndgui_or_nextModel(self, pred_num):
        if pred_num == 0 or pred_num == 1:
            return self.text
        else:
            return "客戶需要安撫，請協助處理。" 

if __name__=="__main__":

    text = test
    sm = sentimentModel(text)
    clear_output()
    model = sm.loadModel(sm.modelPath)

    print(f"要預測的話:\n{text}")
    print("="*25 + "開始預測" + "="*25)
    token_tensor, segment_tensor, mask_tensor = sm.convert_text_to_bertEat(text)
    pred_num, pred = sm.sentimentPredict(model, token_tensor, segment_tensor, mask_tensor)

    sign = sm.pass_sign_to_lineAndgui_or_nextModel(pred_num)
    
    print(f"預測結果為:{pred}")
    print("="*25 + "預測結束" + "="*25)

