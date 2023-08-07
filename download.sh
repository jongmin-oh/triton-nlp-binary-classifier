if [[ -f .env ]]; then
    export $(cat .env | sed 's/#.*//g' | xargs)
fi

echo  "Downloading model from $S3_BUCKET_PATH/$MODEL_FILE"
aws s3 cp $S3_BUCKET_PATH/$MODEL_FILE $PWD/model_repository/curse/1/model.onnx