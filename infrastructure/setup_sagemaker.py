# =============================================================
# setup_sagemaker.py — SageMaker Infrastructure Setup
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Usage: python setup_sagemaker.py
# Prerequisites: AWS CLI configured + boto3 installed
# =============================================================

import boto3
import json
import time

REGION         = 'us-east-1'
BUCKET         = 'fraud-detection-mlproject-armand'
DOMAIN_NAME    = 'fraud-detection-domain'
SPACE_NAME     = 'fraud-detection'
INSTANCE_TYPE  = 'ml.t3.large'

sm = boto3.client('sagemaker', region_name=REGION)
iam = boto3.client('iam', region_name=REGION)


def get_execution_role_arn() -> str:
    """
    Get the SageMaker execution role ARN.

    Returns
    -------
    str : Execution role ARN
    """
    roles = iam.list_roles()['Roles']
    for role in roles:
        if 'SageMaker-ExecutionRole' in role['RoleName']:
            return role['Arn']
    raise ValueError(
        "SageMaker execution role not found!\n"
        "Create a SageMaker domain first in AWS Console."
    )


def add_kinesis_to_role(role_name: str) -> None:
    """
    Add Kinesis permissions to SageMaker execution role.

    Parameters
    ----------
    role_name : str  SageMaker execution role name
    """
    kinesis_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "kinesis:GetRecords",
                    "kinesis:GetShardIterator",
                    "kinesis:DescribeStream",
                    "kinesis:ListStreams",
                    "kinesis:PutRecord",
                    "kinesis:PutRecords",
                    "kinesis:CreateStream",
                    "kinesis:DeleteStream",
                    "kinesis:ListShards"
                ],
                "Resource": "*"
            }
        ]
    }

    iam.put_role_policy(
        RoleName       = role_name,
        PolicyName     = 'KinesisFullAccess',
        PolicyDocument = json.dumps(kinesis_policy)
    )
    print(f"   Kinesis permissions added to {role_name}")


def create_kinesis_stream(
    stream_name: str = 'fraud-detection-stream',
    shard_count: int = 1
) -> dict:
    """
    Create a Kinesis stream for real-time fraud scoring.

    Parameters
    ----------
    stream_name : str  Name of the stream
    shard_count : int  Number of shards (default 1)

    Returns
    -------
    dict : Stream details
    """
    kin = boto3.client('kinesis', region_name=REGION)

    try:
        kin.create_stream(
            StreamName = stream_name,
            ShardCount = shard_count
        )
        print(f"   Creating stream: {stream_name}...")

        # Wait for stream to become ACTIVE
        waiter = kin.get_waiter('stream_exists')
        waiter.wait(StreamName=stream_name)

        desc = kin.describe_stream(
            StreamName=stream_name
        )['StreamDescription']
        print(f"   Stream ACTIVE: {stream_name}")
        print(f"   ARN: {desc['StreamARN']}")

        return {
            'stream_name': stream_name,
            'arn'        : desc['StreamARN'],
            'shards'     : shard_count,
            'cost'       : '$0.015/hour'
        }

    except kin.exceptions.ResourceInUseException:
        print(f"   Stream already exists: {stream_name}")
        return {'stream_name': stream_name}


def delete_kinesis_stream(
    stream_name: str = 'fraud-detection-stream'
) -> None:
    """
    Delete Kinesis stream to stop costs.

    Parameters
    ----------
    stream_name : str  Stream to delete
    """
    kin = boto3.client('kinesis', region_name=REGION)
    try:
        kin.delete_stream(StreamName=stream_name)
        print(f"   Deleted: {stream_name}")
        print(f"   Saving $0.015/hour!")
    except Exception as e:
        print(f"   Error: {e}")


def check_infrastructure() -> None:
    """
    Check current AWS infrastructure status
    and print a summary report.
    """
    print("=" * 55)
    print("  INFRASTRUCTURE STATUS CHECK")
    print("=" * 55)

    # S3 bucket
    s3 = boto3.client('s3', region_name=REGION)
    try:
        s3.head_bucket(Bucket=BUCKET)
        print(f"\n   S3 Bucket     : OK")
        print(f"   Bucket name   : {BUCKET}")

        # List objects
        response = s3.list_objects_v2(
            Bucket=BUCKET, Delimiter='/'
        )
        folders = [
            p['Prefix']
            for p in response.get(
                'CommonPrefixes', []
            )
        ]
        print(f"   Folders       : {folders}")

    except Exception:
        print(f"\n   S3 Bucket     : NOT FOUND")
        print(f"   Run setup_aws.sh first!")

    # SageMaker domains
    try:
        domains = sm.list_domains()['Domains']
        if domains:
            d = domains[0]
            print(f"\n   SageMaker     : OK")
            print(f"   Domain        : {d['DomainName']}")
            print(f"   Status        : {d['Status']}")
        else:
            print(f"\n   SageMaker     : No domains found")
    except Exception as e:
        print(f"\n   SageMaker     : ERROR - {e}")

    # Kinesis streams
    kin = boto3.client('kinesis', region_name=REGION)
    try:
        streams = kin.list_streams()['StreamNames']
        if streams:
            print(f"\n   Kinesis       : {streams}")
            print(f"   Cost          : "
                  f"${len(streams) * 0.015:.3f}/hour")
            print(f"   WARNING       : "
                  f"Delete streams when not in use!")
        else:
            print(f"\n   Kinesis       : No active streams")
    except Exception as e:
        print(f"\n   Kinesis       : ERROR - {e}")

    print(f"\n{'='*55}")


if __name__ == '__main__':
    check_infrastructure()





