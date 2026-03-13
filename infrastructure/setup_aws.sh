#!/bin/bash
# =============================================================
# setup_aws.sh — AWS Infrastructure Setup
# Production Fraud Detection Platform on AWS
# Author: Armand Junior Dongmo Notue
# =============================================================
# Usage: bash setup_aws.sh
# Prerequisites: AWS CLI installed + configured
# =============================================================

set -e  # Exit on any error

BUCKET_NAME="fraud-detection-mlproject-armand"
REGION="us-east-1"
IAM_USER="ml-engineer"

echo "============================================="
echo "  AWS Infrastructure Setup"
echo "  Fraud Detection Platform"
echo "============================================="

# ── STEP 1: Create S3 bucket ──────────────────────
echo ""
echo "Step 1: Creating S3 bucket..."

aws s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$REGION" \
    --create-bucket-configuration \
        LocationConstraint="$REGION" \
    2>/dev/null || echo "   Bucket already exists!"

# Create folder structure
for folder in \
    "raw-data/" \
    "processed-data/" \
    "models/v1/" \
    "models/v2/" \
    "models/v3/" \
    "models/v4/"; do
    aws s3api put-object \
        --bucket "$BUCKET_NAME" \
        --key "$folder" > /dev/null
done

echo "   Bucket created: s3://$BUCKET_NAME"
echo "   Folders: raw-data/, processed-data/, models/"

# ── STEP 2: Create IAM user ───────────────────────
echo ""
echo "Step 2: Creating IAM user..."

aws iam create-user \
    --user-name "$IAM_USER" \
    2>/dev/null || echo "   User already exists!"

aws iam attach-user-policy \
    --user-name "$IAM_USER" \
    --policy-arn \
        "arn:aws:iam::aws:policy/AdministratorAccess" \
    2>/dev/null || echo "   Policy already attached!"

echo "   IAM user created: $IAM_USER"

# ── STEP 3: Create access keys ────────────────────
echo ""
echo "Step 3: Creating access keys..."
echo "   Run: aws iam create-access-key \\"
echo "            --user-name $IAM_USER"
echo "   Save the AccessKeyId and SecretAccessKey!"

# ── STEP 4: Add Kinesis policy ────────────────────
echo ""
echo "Step 4: Kinesis permissions for SageMaker..."

cat > /tmp/kinesis_policy.json << EOF
{
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
EOF

aws iam put-user-policy \
    --user-name "$IAM_USER" \
    --policy-name "KinesisFullAccess" \
    --policy-document file:///tmp/kinesis_policy.json \
    2>/dev/null || echo "   Policy already set!"

echo "   Kinesis permissions added!"

# ── STEP 5: Set budget alert ──────────────────────
echo ""
echo "Step 5: Budget alert..."
echo "   Go to AWS Console → Billing → Budgets"
echo "   Create budget: \$5 zero-spend alert"
echo "   (Cannot be set via CLI without billing access)"

echo ""
echo "============================================="
echo "  Setup Complete!"
echo "  Bucket : s3://$BUCKET_NAME"
echo "  Region : $REGION"
echo "  User   : $IAM_USER"
echo "============================================="




