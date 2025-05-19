import boto3

def read_s3_text_file(bucket_name, key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=key)
    content = response['Body'].read().decode('utf-8')
    return content

if __name__ == "__main__":
    bucket = 'oskar-mlops-results'
    key = 'test_image2_prediction.txt'

    text_content = read_s3_text_file(bucket, key)
    print(text_content)