import json
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )


# Define our input prompt template for the task.
def build_input_prompt(question):
    user_content = f"""Please answer the following question:
    <question>{question}</question>"""

    messages = [{'role': 'user', 'content': [{'text': user_content}]}]
    return messages

# Get completions for each input.
# Using Converse API for unified interface across all models
def get_completion(messages, model="anthropic.claude-3-sonnet-20240229-v1:0", max_tokens=2048, temperature=0.5, top_p=0.9):
    # Use Converse API for unified interface
    # Note: Some models (like Claude) don't allow both temperature and top_p
    # For Claude models, we'll use temperature only
    inference_config = {
        "maxTokens": max_tokens,
        "temperature": temperature
    }
    
    # Only add topP for non-Anthropic models
    if not model.startswith("anthropic.") and not model.startswith("us.anthropic."):
        inference_config["topP"] = top_p
    
    try:
        response = bedrock_runtime.converse(
            modelId=model,
            messages=messages,
            inferenceConfig=inference_config
        )
        
        # Extract text from response
        return response['output']['message']['content'][0]['text']
        
    except Exception as e:
        return f"Error calling model: {str(e)}"


# Get completions for each question in the eval.
# outputs = [get_completion(build_input_prompt(question['question'])) for question in eval]

# # Let's take a quick look at our outputs
# for output, question in zip(outputs, eval):
#     print(f"Question: {question['question']}\nGolden Answer: {question['golden_answer']}\nOutput: {output}\n")