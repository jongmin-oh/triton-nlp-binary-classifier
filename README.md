# triton-nlp-binary-classifier
nvidia triton 을 이용한 NLP binary classifier 모델 서버 구축
 - 준비물 : onnx 모델 <br>
   : (model_respository/curse/{model_name}.onnx 이름으로 존재해야함.)

## Triton?
 - NVIDIA Triton Inference Server는 딥러닝 모델을 호스팅하고 추론을 수행하기 위한 오픈 소스 솔루션입니다.
 - Triton은 GPU 및 CPU에서의 고성능 추론을 지원하며, 모델 배포, 확장성, 동적 모델 로딩, 다양한 프레임워크 지원 등 다양한 기능을 제공합니다.

## 배경
~~~
현재 모델 서빙은 파이썬 웹 프레임워크인 fastapi를 사용해서 구현되어있다.
하지만 모델이 바뀔때마다, 많은 코드 수정이 동반되었고 하나의 컨테이너로 관리되어서 편했지만
python 이라는 고질적인 속도 문제와, 현재는 CPU만 사용하고 있지만 나중에 GPU로 넘어갈때의 용이성(다이나믹배치 등)아
고려되면서 triton 서버를 따로 분리해서 사용하는 게 좋을 것 같다는 생각을 했다.

무엇보다도 카카오페이 블로그에서 비교한 자료를 보고난 후 생각이 바뀌었다.
~~~
[카카오페이 기술 블로그](https://tech.kakaopay.com/post/model-serving-framework/)

### 예시 모델 : 욕설 검출 모델(curse-detection)
 - kcbert를 욕설데이터로 fine-turning한 모델

### triton dokcer image 다운로드
```bash
docker pull nvcr.io/nvidia/tritonserver:23.07-py3
```

### docker run (only-cpu)
```
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:23.07-py3 tritonserver --model-repository=/model
```

***

### request
```python
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

```
