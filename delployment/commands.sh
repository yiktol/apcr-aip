
Bucket='yikyakyuk-ap-southeast-1-875692608981'
Prefix='templates/demo'
BucketRegion='ap-southeast-1'

# Upload templates to S3
aws s3 cp ./ s3://$Bucket/$Prefix --recursive

# Deploy Stack to Singapore Region
aws cloudformation deploy \
    --region ap-southeast-1 \
    --stack-name genai \
    --template-file master2.yaml \
    --parameter-overrides Bucket=$Bucket Prefix=$Prefix BucketRegion=$BucketRegion \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND

aws cloudformation deploy \
    --region ap-southeast-1 \
    --stack-name genai \
    --s3-bucket $Bucket \
    --s3-prefix $Prefix  \
    --template-file master2.yaml \
    --parameter-overrides Bucket=$Bucket Prefix=$Prefix BucketRegion=$BucketRegion \
    --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND


# Deploy Stack to Singapore Region
aws cloudformation create-stack \
    --region ap-southeast-1 \
    --stack-name master \
    --template-url https://$Bucket.s3-$BucketRegion.amazonaws.com/$Prefix/master2.yaml \
    --parameters ParameterKey=Bucket,ParameterValue=$Bucket \
                 ParameterKey=Prefix,ParameterValue=$Prefix \
                 ParameterKey=BucketRegion,ParameterValue=$BucketRegion \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND

# Deploy Stack to Singapore Region
aws cloudformation update-stack \
    --region ap-southeast-1 \
    --stack-name master \
    --template-url https://$Bucket.s3-$BucketRegion.amazonaws.com/$Prefix/master2.yaml \
    --parameters ParameterKey=Bucket,ParameterValue=$Bucket \
                 ParameterKey=Prefix,ParameterValue=$Prefix \
                 ParameterKey=BucketRegion,ParameterValue=$BucketRegion \
    --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND


