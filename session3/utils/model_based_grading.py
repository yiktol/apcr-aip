import json
import re
import streamlit as st
import boto3

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1', 
    )

# We start by defining a "grader prompt" template.
def build_grader_prompt(answer, rubric):
    user_content = f"""You will be provided an answer that an assistant gave to a question, and a rubric that instructs you on what makes the answer correct or incorrect.
    
    Here is the answer that the assistant gave to the question.
    <answer>{answer}</answer>
    
    Here is the rubric on what makes the answer correct or incorrect.
    <rubric>{rubric}</rubric>
    
    An answer is correct if it entirely meets the rubric criteria, and is otherwise incorrect. =
    First, think through whether the answer is correct or incorrect based on the rubric inside <thinking></thinking> tags. Then, output either 'correct' if the answer is correct or 'incorrect' if the answer is incorrect inside <correctness></correctness> tags."""

    messages = [{'role': 'user', 'content': [{'text': user_content}]}]
    return messages

# Now we define the full grade_completion function.


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


def grade_completion(output, golden_answer, grader_model="anthropic.claude-3-sonnet-20240229-v1:0"):
    messages = build_grader_prompt(output, golden_answer)
    completion = get_completion(messages, model=grader_model)
    st.info(completion)
    # Extract just the label from the completion (we don't care about the thinking)
    pattern = r'<correctness>(.*?)</correctness>'
    match = re.search(pattern, completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Did not find <correctness></correctness> tags.")