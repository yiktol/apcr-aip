# Selectbox Update Issue - Fix Pattern

## Problem
When a selectbox changes, text_area widgets don't update because they have static keys and use the `value=` parameter.

## Proper Streamlit Pattern

According to [Streamlit documentation](https://docs.streamlit.io/knowledge-base/using-streamlit/widget-updating-session-state):

1. Use a `key` parameter on the widget
2. Update `st.session_state[key]` in a callback function
3. Don't use `value=` parameter - let Streamlit manage it via the key

## Fix Pattern

### BEFORE (Broken):
```python
example_prompts = {
    "Option 1": "Prompt text 1",
    "Option 2": "Prompt text 2",
    "Custom": "Write your own..."
}

selected_example = st.selectbox(
    "Choose an example",
    options=list(example_prompts.keys()),
    key="my_examples"
)

if selected_example == "Custom":
    user_prompt = st.text_area(
        "Your prompt",
        value="",
        height=120,
        key="my_custom"
    )
else:
    user_prompt = example_prompts[selected_example]
    if user_prompt:
        st.text_area(
            "Prompt to test",
            value=user_prompt,  # ❌ This doesn't update!
            height=120,
            key="my_selected"
        )
```

### AFTER (Fixed):
```python
example_prompts = {
    "Option 1": "Prompt text 1",
    "Option 2": "Prompt text 2",
    "Custom": "Write your own..."
}

# Initialize session state
if 'my_prompt' not in st.session_state:
    st.session_state.my_prompt = ""

# Callback for selectbox change
def on_example_change():
    selected = st.session_state.my_examples
    if selected == "Custom":
        st.session_state.my_prompt = ""
    else:
        st.session_state.my_prompt = example_prompts[selected]

selected_example = st.selectbox(
    "Choose an example",
    options=list(example_prompts.keys()),
    key="my_examples",
    on_change=on_example_change  # ✅ Callback updates session state
)

if selected_example == "Custom":
    user_prompt = st.text_area(
        "Your prompt",
        height=120,
        placeholder="Enter your custom prompt...",
        key="my_prompt"  # ✅ Same key as session state
    )
else:
    user_prompt = st.text_area(
        "Prompt to test",
        height=120,
        key="my_prompt"  # ✅ Same key, no value parameter
    )
```

## Key Points

1. **Single text_area key**: Use the same key for both Custom and non-Custom cases
2. **No value parameter**: Remove `value=` from text_area - Streamlit manages it via key
3. **Callback function**: Update `st.session_state[key]` in the callback
4. **Initialize session state**: Set initial value if key doesn't exist

## Pages That Need Fixing

### ✅ Fixed:
- `03_CoT_and_ToT.py` - All 3 tabs (Zero-Shot, Few-Shot, ToT)
- `02_Prompt_Engineering_Technique.py` - Few-shot and Few-shot CoT sections
- `00_Elements_of_Prompts.py` - Instruction and Context sections

### ❌ Still Need Fixing:
- `00_Elements_of_Prompts.py` - Input Text, Input Data, Problem/Task, Output Format sections
- `05_LLM_Security_&_Vulnerability.py` - All 7 sections (Injection, Leaking, Jailbreaking, Factuality, Bias, PII, Data Poisoning)
- `31_Bedrock_Guardrails.py` - Sample prompts section

## Testing
After applying the fix:
1. Refresh the browser
2. Select different options from the selectbox
3. The text_area should update immediately
4. Manual edits to text_area should still work
