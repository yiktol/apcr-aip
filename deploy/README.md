# Deployment Configuration

## Security Notice

**IMPORTANT**: Never commit OAuth secrets or API keys to version control.

## Setup Instructions

You have two options for managing secrets:

### Option 1: AWS Systems Manager Parameter Store (Recommended)

Store secrets securely in AWS SSM Parameter Store.

#### 1. Store Parameters in SSM

```bash
cd delployment
./put-parameters.sh --region us-east-1
```

This will prompt you for all required values and store them securely in SSM Parameter Store with:
- Regular parameters stored as `String` type
- Secrets (client secrets, app secrets) stored as `SecureString` type (encrypted with KMS)
- Tagged with `Application=GenAI-Essentials` and `Environment=Production`

#### 2. Deploy Using SSM Template

```bash
aws cloudformation create-stack \
  --stack-name genai-cognito \
  --template-body file://cognito-ssm.yaml \
  --parameters ParameterKey=ParameterPrefix,ParameterValue=/genai/cognito \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND
```

The CloudFormation template will automatically retrieve parameters from SSM during deployment.

#### 3. Update Stack

```bash
aws cloudformation update-stack \
  --stack-name genai-cognito \
  --template-body file://cognito-ssm.yaml \
  --parameters ParameterKey=ParameterPrefix,ParameterValue=/genai/cognito \
  --capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND
```

### Option 2: Local Parameters File

Use a local JSON file for parameters (not recommended for production).

#### 1. Create Parameters File

```bash
cp cognito-parameters.example.json cognito-parameters.json
```

Edit `cognito-parameters.json` with your actual OAuth credentials.

#### 2. Deploy CloudFormation Stack

```bash
aws cloudformation create-stack \
  --stack-name genai-cognito \
  --template-body file://cognito.yaml \
  --parameters file://cognito-parameters.json \
  --capabilities CAPABILITY_IAM
```

#### 3. Update Stack

```bash
aws cloudformation update-stack \
  --stack-name genai-cognito \
  --template-body file://cognito.yaml \
  --parameters file://cognito-parameters.json \
  --capabilities CAPABILITY_IAM
```

## Utility Scripts

### Store Parameters in SSM
```bash
./put-parameters.sh [--profile PROFILE] [--region REGION]
```

### Retrieve Parameters from SSM
```bash
./get-parameters.sh [--profile PROFILE] [--region REGION] [--output FILE]
```

This generates a `cognito-parameters.json` file from SSM for use with the standard template.

### List All Parameters
```bash
aws ssm get-parameters-by-path \
  --path /genai/cognito \
  --region us-east-1
```

### Get Single Parameter (with decryption)
```bash
aws ssm get-parameter \
  --name /genai/cognito/GoogleClientSecret \
  --with-decryption \
  --region us-east-1
```

## Files

- `cognito.yaml` - Standard CloudFormation template (requires parameters file)
- `cognito-ssm.yaml` - SSM-integrated template (retrieves from Parameter Store)
- `cognito-parameters.example.json` - Example parameters (safe to commit)
- `cognito-parameters.json` - Actual parameters with secrets (NEVER commit - in .gitignore)
- `put-parameters.sh` - Script to store parameters in SSM
- `get-parameters.sh` - Script to retrieve parameters from SSM

## OAuth Provider Setup

### Google OAuth
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create OAuth 2.0 credentials
3. Add authorized redirect URIs matching your Cognito domain

### Facebook Login
1. Go to [Facebook Developers](https://developers.facebook.com/)
2. Create a new app
3. Configure OAuth redirect URIs

### Amazon Login
1. Go to [Amazon Developer Console](https://developer.amazon.com/)
2. Create a Security Profile
3. Configure allowed return URLs

## Security Best Practices

1. **Use SSM Parameter Store** for production deployments
2. **Enable KMS encryption** for SecureString parameters (automatic with SSM)
3. **Rotate credentials regularly** using the `put-parameters.sh` script
4. **Use IAM policies** to restrict access to SSM parameters
5. **Never commit** `cognito-parameters.json` to version control
6. **Use AWS Secrets Manager** for application runtime secrets (already configured in template)
