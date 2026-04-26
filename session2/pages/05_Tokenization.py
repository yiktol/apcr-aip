import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import uuid
import re
import json
from collections import Counter
from transformers import AutoTokenizer
from utils.styles import load_css, custom_header, sub_header, create_footer, AWS_COLORS
from utils.common import render_sidebar
import utils.authenticate as authenticate


st.set_page_config(
    page_title="Tokenization",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "bpe_step" not in st.session_state:
        st.session_state.bpe_step = 0
    if "bpe_merges" not in st.session_state:
        st.session_state.bpe_merges = []


@st.cache_resource
def load_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ── Tab 1: Live Tokenizer ──────────────────────────────────────────

def live_tokenizer_tab():
    st.markdown(sub_header("Live Tokenizer", "🔤", "aws"), unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Type any text and see how GPT-2's BPE tokenizer breaks it into subword tokens with their IDs.
    </div>""", unsafe_allow_html=True)

    tokenizer = load_tokenizer("gpt2")

    text = st.text_area("Enter text:", "A puppy is to dog as kitten is to cat.", height=100)

    if not text:
        return

    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text)

    # Token chips
    colors = ["#FF9900", "#0073BB", "#3EB489", "#D13212", "#9B59B6", "#16B9D4", "#E67E22", "#2ECC71"]
    chips = "<div class='card'>"
    for i, (tok, tid) in enumerate(zip(tokens, token_ids)):
        bg = colors[i % len(colors)]
        display_tok = tok.replace("Ġ", "▁")
        chips += (
            f"<span style='background:{bg}; color:white; padding:6px 12px; margin:3px; "
            f"border-radius:6px; display:inline-block; font-family:monospace;'>"
            f"{display_tok} <small style=\"opacity:0.7\">({tid})</small></span>"
        )
    st.markdown(chips, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tokens", len(tokens))
    col2.metric("Characters", len(text))
    col3.metric("Ratio", f"{len(text)/max(len(tokens),1):.1f} chars/token")


# ── Tab 2: Tokenization Strategies ─────────────────────────────────

def strategies_tab():
    st.markdown(sub_header("Tokenization Strategies", "🧪", "aws"), unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Compare word-level, character-level, and subword (BPE) tokenization on the same input.
    </div>""", unsafe_allow_html=True)

    text = st.text_input("Enter text:", "A puppy is to dog as kitten is to cat.", key="strat_input")

    if not text:
        return

    # Word-level
    word_tokens = text.split()

    # Character-level
    char_tokens = list(text)

    # Subword (BPE)
    tokenizer = load_tokenizer("gpt2")
    subword_tokens = tokenizer.tokenize(text)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Word-level**")
        st.metric("Count", len(word_tokens))
        for t in word_tokens:
            st.code(t, language=None)

    with col2:
        st.markdown("**Character-level**")
        st.metric("Count", len(char_tokens))
        st.code(" | ".join(char_tokens), language=None)

    with col3:
        st.markdown("**Subword (BPE)**")
        st.metric("Count", len(subword_tokens))
        for t in subword_tokens:
            st.code(t.replace("Ġ", "▁"), language=None)

    # Comparison chart
    fig = go.Figure(data=[go.Bar(
        x=["Word", "Character", "Subword (BPE)"],
        y=[len(word_tokens), len(char_tokens), len(subword_tokens)],
        marker_color=[AWS_COLORS["primary"], AWS_COLORS["danger"], AWS_COLORS["accent"]],
        text=[len(word_tokens), len(char_tokens), len(subword_tokens)],
        textposition="outside",
    )])
    fig.update_layout(title="Token Count Comparison", yaxis_title="Tokens", height=350)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 3: Compare Models ──────────────────────────────────────────

def compare_models_tab():
    st.markdown(sub_header("Compare Tokenizers Across Models", "🔀", "aws"), unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    See how GPT-2, BERT, and T5 tokenize the same text differently — affecting token counts and context usage.
    </div>""", unsafe_allow_html=True)

    text = st.text_area(
        "Enter text:",
        "A puppy is to dog as kitten is to cat.",
        height=80,
        key="compare_input",
    )

    if not text:
        return

    models = {
        "GPT-2 (BPE)": "gpt2",
        "BERT (WordPiece)": "bert-base-uncased",
        "T5 (SentencePiece)": "t5-small",
    }

    results = {}
    for label, name in models.items():
        tok = load_tokenizer(name)
        tokens = tok.tokenize(text)
        results[label] = {
            "tokens": tokens,
            "count": len(tokens),
            "vocab_size": tok.vocab_size,
        }

    # Side-by-side
    cols = st.columns(len(models))
    for col, (label, data) in zip(cols, results.items()):
        with col:
            st.markdown(f"**{label}**")
            st.metric("Tokens", data["count"])
            st.metric("Vocab Size", f"{data['vocab_size']:,}")
            display = [t.replace("Ġ", "▁").replace("##", "##") for t in data["tokens"]]
            st.code(" ".join(display), language=None)

    # Grouped bar chart
    fig = go.Figure(data=[
        go.Bar(name="Token Count", x=list(results.keys()),
               y=[d["count"] for d in results.values()],
               marker_color=AWS_COLORS["accent"],
               text=[d["count"] for d in results.values()],
               textposition="outside"),
    ])
    fig.update_layout(title="Token Counts by Model", yaxis_title="Tokens", height=350)
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 4: BPE Merge Explorer ──────────────────────────────────────

def _get_pair_freqs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def _merge_pair(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab


def bpe_explorer_tab():
    st.markdown(sub_header("BPE Merge Explorer", "🎮", "aws"), unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Step through the Byte-Pair Encoding algorithm — watch how frequent character pairs merge into subword tokens.
    </div>""", unsafe_allow_html=True)

    corpus_text = st.text_area(
        "Corpus:",
        "low lower newest widest low lower lower newest newest newest widest",
        height=68,
        key="bpe_corpus",
    )

    if not corpus_text:
        return

    # Build initial vocab
    words = corpus_text.strip().split()
    word_freqs = Counter(words)
    initial_vocab = {}
    for word, freq in word_freqs.items():
        spaced = " ".join(list(word)) + " </w>"
        initial_vocab[spaced] = freq

    # Compute all merges up front
    max_merges = 15
    vocab = dict(initial_vocab)
    all_steps = [{"vocab": dict(vocab), "merge": None}]

    for _ in range(max_merges):
        pairs = _get_pair_freqs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = _merge_pair(best, vocab)
        all_steps.append({"vocab": dict(vocab), "merge": best, "freq": pairs[best]})

    total_merges = len(all_steps) - 1

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("⏭️ Next Merge", disabled=st.session_state.bpe_step >= total_merges):
            st.session_state.bpe_step += 1
    with col2:
        if st.button("🔄 Reset"):
            st.session_state.bpe_step = 0

    step = min(st.session_state.bpe_step, total_merges)
    current = all_steps[step]

    st.markdown(f"**Step {step} / {total_merges}**")

    if current["merge"]:
        a, b = current["merge"]
        st.success(f"Merged: `{a}` + `{b}` → `{a}{b}` (frequency: {current['freq']})")

    # Show current vocabulary
    st.markdown("**Current Vocabulary:**")
    vocab_df = pd.DataFrame(
        [{"Token Sequence": k, "Frequency": v} for k, v in current["vocab"].items()]
    )
    st.dataframe(vocab_df, use_container_width=True, hide_index=True)

    # Vocabulary size over steps
    vocab_sizes = []
    for s in all_steps[: step + 1]:
        unique_tokens = set()
        for word in s["vocab"]:
            unique_tokens.update(word.split())
        vocab_sizes.append(len(unique_tokens))

    fig = go.Figure(data=go.Scatter(
        x=list(range(len(vocab_sizes))),
        y=vocab_sizes,
        mode="lines+markers",
        marker=dict(color=AWS_COLORS["primary"]),
        line=dict(color=AWS_COLORS["accent"]),
    ))
    fig.update_layout(
        title="Vocabulary Size Over Merges",
        xaxis_title="Merge Step",
        yaxis_title="Unique Tokens",
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Tab 5: Bedrock Token Costs ──────────────────────────────────────

def bedrock_cost_tab():
    st.markdown(sub_header("Bedrock Token Costs", "💰", "aws"), unsafe_allow_html=True)
    st.markdown("""<div class='info-box'>
    Send a prompt to Amazon Bedrock and see real input/output token counts with estimated cost.
    </div>""", unsafe_allow_html=True)

    try:
        import boto3
        bedrock = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    except Exception as e:
        st.warning(f"Could not connect to Bedrock: {e}")
        st.info("This demo requires AWS credentials with Bedrock access in us-east-1.")
        return

    model_options = {
        "Amazon Nova Lite": "amazon.nova-lite-v1:0",
        "Amazon Nova Micro": "amazon.nova-micro-v1:0",
    }

    col1, col2 = st.columns([2, 1])
    with col1:
        prompt = st.text_area("Enter a prompt:", "A puppy is to dog as kitten is to cat.", height=80, key="bedrock_prompt")
    with col2:
        model_label = st.selectbox("Model:", list(model_options.keys()))
        max_tokens = st.slider("Max output tokens:", 50, 500, 200, 50)

    if st.button("🚀 Send to Bedrock", type="primary"):
        model_id = model_options[model_label]

        with st.spinner("Calling Bedrock..."):
            try:
                response = bedrock.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7},
                )

                output_text = response["output"]["message"]["content"][0]["text"]
                usage = response["usage"]
                input_tokens = usage["inputTokens"]
                output_tokens = usage["outputTokens"]

                st.markdown("**Response:**")
                st.markdown(f"<div class='card'>{output_text}</div>", unsafe_allow_html=True)

                st.markdown("---")

                c1, c2, c3 = st.columns(3)
                c1.metric("Input Tokens", input_tokens)
                c2.metric("Output Tokens", output_tokens)
                c3.metric("Total Tokens", input_tokens + output_tokens)

                # Cost estimation (approximate pricing)
                pricing = {
                    "Amazon Nova Lite": {"input": 0.00006, "output": 0.00024},
                    "Amazon Nova Micro": {"input": 0.000035, "output": 0.00014},
                }
                prices = pricing.get(model_label, {"input": 0.0001, "output": 0.0004})
                input_cost = (input_tokens / 1000) * prices["input"]
                output_cost = (output_tokens / 1000) * prices["output"]
                total_cost = input_cost + output_cost

                st.markdown(f"**Estimated cost:** ${total_cost:.6f} "
                            f"(input: ${input_cost:.6f} + output: ${output_cost:.6f})")

            except Exception as e:
                st.error(f"Bedrock error: {e}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    initialize_session_state()
    load_css()

    with st.sidebar:
        render_sidebar()

        with st.expander("About this App", expanded=False):
            st.markdown("""
            This app demonstrates how tokenization works in transformer models.

            **Topics Covered:**
            - Live subword tokenization with GPT-2
            - Word vs character vs subword strategies
            - Cross-model tokenizer comparison
            - BPE merge algorithm walkthrough
            - Bedrock token cost estimation
            """)

    st.markdown(custom_header("🔤 Tokenization", 1), unsafe_allow_html=True)

    st.markdown("""<div class='info-box'>
    Tokenization is the first step in any transformer pipeline — converting raw text into numerical tokens
    that models can process. Different models tokenize differently, directly affecting cost, context limits, and performance.
    </div>""", unsafe_allow_html=True)

    tabs = st.tabs([
        "🔤 Live Tokenizer",
        "🧪 Strategies",
        "🔀 Compare Models",
        "🎮 BPE Explorer",
        "💰 Bedrock Costs",
    ])

    with tabs[0]:
        live_tokenizer_tab()
    with tabs[1]:
        strategies_tab()
    with tabs[2]:
        compare_models_tab()
    with tabs[3]:
        bpe_explorer_tab()
    with tabs[4]:
        bedrock_cost_tab()

    create_footer()


if __name__ == "__main__":
    try:
        if "localhost" in st.context.headers.get("host", "localhost"):
            main()
        else:
            is_authenticated = authenticate.login()
            if is_authenticated:
                main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        with st.expander("Error Details"):
            st.code(str(e))
