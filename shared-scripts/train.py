import os
import boto3
import tempfile
import json
import shutil
from arguments import parse_args
import train_lib
import time

os.environ["NVTE_TORCH_COMPILE"] = "0"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "**********************" # token here

def download_model_from_s3(bucket_name, s3_model_path, local_path, max_retries=3):
    s3 = boto3.client('s3')
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    s3_objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=s3_model_path)
    for obj in s3_objects.get('Contents', []):
        s3_file_path = obj['Key']
        local_file_path = os.path.join(local_path, os.path.relpath(s3_file_path, s3_model_path))
        local_file_dir = os.path.dirname(local_file_path)

        if not os.path.exists(local_file_dir):
            os.makedirs(local_file_dir)

        retries = 0
        while retries < max_retries:
            try:
                # Use a unique temp file for each download
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    temp_file_path = tmp_file.name

                s3.download_file(bucket_name, s3_file_path, temp_file_path)
                shutil.move(temp_file_path, local_file_path)
                break  # Break if download is successful
            except Exception as e:
                retries += 1
                print(f"Failed to download {s3_file_path}. Retry {retries}/{max_retries}. Error: {e}")
                time.sleep(2 ** retries)  # Exponential backoff
                if retries == max_retries:
                    raise

def verify_model_files(local_path):
    expected_files = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json"
    ]

    missing_files = []
    for file in expected_files:
        if not os.path.exists(os.path.join(local_path, file)):
            missing_files.append(file)

    if missing_files:
        raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
    else:
        print("All expected model files are present.")

def update_config_path(local_path):
    config_path = os.path.join(local_path, 'config.json')
    
    # Validate JSON content
    with open(config_path, 'r') as f:
        content = f.read()
        if not content.strip():
            raise ValueError(f"{config_path} is empty")
        config = json.loads(content)
    
    config['_name_or_path'] = local_path
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Updated config.json _name_or_path to {local_path}")

def print_model_files(local_path):
    print(f"Model files located at {local_path}:")
    for root, dirs, files in os.walk(local_path):
        for file in files:
            print(os.path.join(root, file))

def main():
    """Main function to train GPT."""
    args, _ = parse_args()
    
    # Download the model from S3 to the local path
    # s3_model_path = 'model-v0'  # The path to your model in the S3 bucket
    # local_model_path = '/opt/ml/input/data/model'  # Local path on the training instance
    
    # print(f"Downloading model from s3://{s3_bucket_name}/{s3_model_path} to {local_model_path}")
    # download_model_from_s3(s3_bucket_name, s3_model_path, local_model_path)
    
    # Verify that all expected model files are present
    # verify_model_files(local_model_path)
    
    # Update the config.json to point to the local model path
    # update_config_path(local_model_path)
    
    # Print the locations of the model files
    # print_model_files(local_model_path)
    
    # Update the args to point to the local model path
    # args.hf_pretrained_model_name_or_dir = local_model_path
    
    train_lib.main(args)

if __name__ == "__main__":
    
    main()
