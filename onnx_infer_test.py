from transformers import AutoTokenizer
import numpy as np
import onnxruntime as rt

session = rt.InferenceSession(
    "model_repository/curse/1/model.onnx",
    providers=["CPUExecutionProvider"],
)

tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")


def infer(sentence: str):
    sentence = tokenizer(
        sentence,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="np",
    )

    input_feed = {
        "input_ids": sentence["input_ids"],
        "attention_mask": sentence["attention_mask"],
        "token_type_ids": sentence["token_type_ids"],
    }

    out = session.run(input_feed=input_feed, output_names=["output_0"])[0]
    print(out[0])
    softmax = np.exp(out) / np.sum(np.exp(out), axis=-1, keepdims=True)
    return softmax[0]


if __name__ == "__main__":
    print(infer("씨발")[1])
