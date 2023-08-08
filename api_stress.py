from typing import Final
import subprocess

from locust import HttpUser, task, between
from transformers import AutoTokenizer

host = "http://localhost:8000"

MAX_LENGTH: Final[int] = 128
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")


class MyUser(HttpUser):
    wait_time = between(3, 4)

    @task
    def post_request(self):
        headers = {"Content-Type": "application/json"}
        bert_inputs = tokenizer(
            "안녕하세요 감사해요 잘있어요 다시만나요",
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="np",
        )
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
        # print(data["context"])

        self.client.post(
            "/v2/models/curse/versions/1/infer", headers=headers, json=data
        )


if __name__ == "__main__":
    subprocess.run(
        ["locust", "-f", "api_stress.py", "--host=" + host],
        check=True,
    )
