from typing import Final
import asyncio

import numpy as np

from tritonclient.http import InferenceServerClient, InferInput
from transformers import AutoTokenizer

MAX_LENGTH: Final[int] = 128


class CurseDetector:
    def __init__(self):
        self.client = None
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        self.url = "localhost:8000"

    async def __aenter__(self):
        self.client = InferenceServerClient(url=self.url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    async def _request(self, sentence: str) -> np.ndarray:
        bert_inputs = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )

        input_data = [
            InferInput("input_ids", bert_inputs["input_ids"].shape, "INT64"),
            InferInput("attention_mask", bert_inputs["attention_mask"].shape, "INT64"),
            InferInput("token_type_ids", bert_inputs["token_type_ids"].shape, "INT64"),
        ]

        for i, name in enumerate(["input_ids", "attention_mask", "token_type_ids"]):
            input_data[i].set_data_from_numpy(bert_inputs[name])

        # async_stream_infer 메소드를 사용하여 비동기 요청 보내기
        output = self.client.async_infer(
            model_name="curse",
            inputs=input_data,
            headers={"Content-Type": "application/json"},
        )

        # future 객체의 result 메소드를 사용하여 응답 기다리기
        output = output.get_result()
        return output.as_numpy("output_0")[0]

    async def predict(self, sentence: str):
        logit = await self._request(sentence)
        softmax = np.exp(logit) / np.sum(np.exp(logit), axis=-1, keepdims=True)
        return softmax[1]


if __name__ == "__main__":

    async def run_prediction():
        async with CurseDetector() as detector:
            result = await detector.predict("씨발")
            print(result)

    asyncio.run(run_prediction())
