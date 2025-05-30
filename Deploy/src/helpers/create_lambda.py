import boto3

REGION = "eu-central-1"

lambda_client = boto3.client('lambda', region_name=REGION)
ecr_client = boto3.client("ecr", region_name=REGION)
ecr_client = boto3.client('ecr', region_name=REGION)
sts_client = boto3.client('sts')


def get_latest_image(repository_name, region=REGION):
    account_id = sts_client.get_caller_identity()['Account']

    response = ecr_client.describe_images(repositoryName=repository_name)
    images = response['imageDetails']
    latest_image = max(images, key=lambda x: x['imagePushedAt'])

    image_tags = latest_image.get('imageTags', [])
    image_digest = latest_image['imageDigest']

    if image_tags:
        image_tag = image_tags[0]
        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{image_tag}"
    else:
        image_uri = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}@{image_digest}"

    return {
        'imageDigest': image_digest,
        'imageTags': image_tags,
        'pushedAt': latest_image['imagePushedAt'],
        'imageUri': image_uri
    }

def create_lambda_function(function_name, image_uri, role_arn):
    response = lambda_client.create_function(
        FunctionName=function_name,
        Role=role_arn,
        Code={
            'ImageUri': image_uri
        },
        PackageType='Image',
        Publish=True,
        Timeout=10,
        MemorySize=1024
    )
    print(f"Lambda function '{function_name}' created.")
    return response


if __name__ == "__main__":
    repository = "kotest-repo"
    
    latest_image = get_latest_image(repository)
    print(f"Latest Image Digest: {latest_image['imageDigest']}")
    print(f"Tags: {latest_image['imageTags']}")
    print(f"Uploaded At: {latest_image['pushedAt']}")
    print(f"Image URI: {latest_image['imageUri']}")
    image_uri = latest_image['imageUri']

    # Replace with your IAM Role ARN and ECR Image URI
    role_arn = 'arn:aws:iam::694676321750:role/lambda-kotest-role'

    create_lambda_function('kotest-lambda-predict', image_uri, role_arn=role_arn)