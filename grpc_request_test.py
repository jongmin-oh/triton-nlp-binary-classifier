from typing import Final
import numpy as np
from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
from transformers import AutoTokenizer

MAX_LENGTH: Final[int] = 128


class CurseDetector:
    def __init__(self):
        self.client = None
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.url = "localhost:8001"  # gRPC는 주로 8001 포트를 사용합니다.

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

        output = InferRequestedOutput("output_0")

        # 추론 요청 보내기
        results = self.client.infer(
            model_name="curse",
            inputs=input_data,
            outputs=[output],
        )

        return results.as_numpy("output_0")[0]

    def predict(self, sentence: str):
        logit = self._request(sentence)
        softmax = np.exp(logit) / np.sum(np.exp(logit), axis=-1, keepdims=True)
        return softmax[1]


if __name__ == "__main__":
    with CurseDetector() as detector:
        print(detector.predict("씨발"))
