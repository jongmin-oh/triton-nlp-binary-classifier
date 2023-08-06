import json
from typing import Final

import requests

import numpy as np
from transformers import AutoTokenizer

MAX_LENGTH: Final[int] = 128

# 요청 URL 설정
url = "http://localhost:8000/v2/models/curse/versions/1/infer"
headers = {"Content-Type": "application/json"}


class CurseDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.timeout = 5

    def _request_triton(self, sentence: str):
        bert_inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        # 입력 데이터 준비
        data = {
            "inputs": [
                {
                    "name": "input_ids",
                    "shape": [1, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": bert_inputs["input_ids"].tolist(),
                },
                {
                    "name": "attention_mask",
                    "shape": [1, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": bert_inputs["attention_mask"].tolist(),
                },
                {
                    "name": "token_type_ids",
                    "shape": [1, MAX_LENGTH],
                    "datatype": "INT64",
                    "data": bert_inputs["token_type_ids"].tolist(),
                },
            ]
        }

        # 요청
        response = requests.post(
            url, headers=headers, data=json.dumps(data), timeout=self.timeout
        )

        # 응답 확인
        if response.status_code == 200:
            output_data = response.json()
            return output_data
        else:
            print("오류 발생:", response.status_code, response.text)

    def predict(self, sentence: str):
        response = self._request_triton(sentence)
        logit = response["outputs"][0]["data"]
        softmax = np.exp(logit) / np.sum(np.exp(logit), axis=-1, keepdims=True)
        return softmax[1]


if __name__ == "__main__":
    detector = CurseDetector()
    print(detector.predict("씨발"))