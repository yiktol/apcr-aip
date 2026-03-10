# Deployment Configuration

## Security Notice

**IMPORTANT**: Never commit OAuth secrets or API keys to version control.

## Setup Instructions

### 1. Create Parameters File

Copy the example parameters file and fill in your actual values:

```bash
cp cognito-parameters.example.json cognito-parameters.json
```

Edit `cognito-parameters.json` with your actual OAuth credentials:
- Google OAuth credentials from [Google Cloud Console](https://console.cloud.google.com/)
- Facebook App credentials from [Facebook Developers](https://developers.facebook.com/)
- Amazon Login credentials from [Amazon Developer Console](https://developer.amazon.com/)

### 2. Deploy CloudFormation Stack

```bash
aws cloudformation create-stack \
  --stack-name genai-cognito \
  --template-body file://cognito.yaml \
  --parameters file://cognito-parameters.json \
  --capabilities CAPABILITY_IAM
```

Or update existing stack:

```bash
aws cloudformation update-stack \
  --stack-name genai-cognito \
  --template-body file://cognito.yaml \
  --parameters file://cognito-parameters.json \
  --capabilities CAPABILITY_IAM
```

## Files

- `cognito.yaml` - CloudFormation template (safe to commit)
- `cognito-parameters.example.json` - Example parameters (safe to commit)
- `cognito-parameters.json` - Actual parameters with secrets (NEVER commit - in .gitignore)

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
