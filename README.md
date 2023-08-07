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
python 이라는 고질적인 속도 문제와, 현재는 CPU만 사용하고 있지만
나중에 GPU로 넘어갈때의 용이성(다이나믹배치 등)이 고려되면서
triton 서버를 따로 분리해서 사용하는게 좋을 것 같다는 생각을 했다.

무엇보다도 카카오페이 블로그에서 비교한 자료를 보고난 후 생각이 바뀌었다.
~~~
[카카오페이 기술 블로그](https://tech.kakaopay.com/post/model-serving-framework/)

### 예시 모델 : 욕설 검출 모델(curse-detection)
 - kcbert를 욕설데이터로 fine-turning한 모델

### triton dokcer image pull
```bash
docker pull nvcr.io/nvidia/tritonserver:23.07-py3
```

### docker run (only-cpu)
```bash
docker run --rm \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v ${PWD}/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.07-py3 \
  tritonserver --model-repository=/model
```

***

### request
```python
from typing import Final
from tritonclient.http import InferenceServerClient
from tritonclient.http import InferInput

import numpy as np

from transformers import AutoTokenizer

MAX_LENGTH: Final[int] = 128


class CurseDetector:
    def __init__(self):
        self.client = None
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.url = "localhost:8000"

    def __enter__(self):
        self.client = InferenceServerClient(url=self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def _request(self, sentence: str) -> np.ndarray:
        bert_inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
        # 입력 데이터 생성
        input_data = [
            InferInput("input_ids", bert_inputs["input_ids"].shape, "INT64"),
            InferInput("attention_mask", bert_inputs["attention_mask"].shape, "INT64"),
            InferInput("token_type_ids", bert_inputs["token_type_ids"].shape, "INT64"),
        ]

        # 입력 데이터에 값을 설정
        for i, name in enumerate(["input_ids", "attention_mask", "token_type_ids"]):
            input_data[i].set_data_from_numpy(bert_inputs[name])

        # 추론 요청 보내기
        output = self.client.infer(
            model_name="curse",
            inputs=input_data,
            headers={"Content-Type": "application/json"},
        )
        return output.as_numpy("output_0")[0]

    def predict(self, sentence: str):
        logit = self._request(sentence)
        softmax = np.exp(logit) / np.sum(np.exp(logit), axis=-1, keepdims=True)
        return softmax[1]


if __name__ == "__main__":
    with CurseDetector() as detector:
        print(detector.predict("씨발")) # 0.98042375
```
