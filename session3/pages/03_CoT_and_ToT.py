
# Refactored Chain-of-Thought Prompting App

import streamlit as st
import logging
import boto3
from botocore.exceptions import ClientError
import uuid
import utils.common as common
import utils.authenticate as authenticate
from utils.styles import load_css
# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Page configuration with improved styling
st.set_page_config(
    page_title="Chain-of-Thought Prompting",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

common.initialize_session_state()

# ------- API FUNCTIONS -------

def text_conversation(bedrock_client, model_id, system_prompts, messages, **params):
    """Sends messages to a model."""
    logger.info(f"Generating message with model {model_id}")
    
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=messages,
            system=system_prompts,
            inferenceConfig=params,
            additionalModelRequestFields={}
        )
        
        # Log token usage
        token_usage = response['usage']
        logger.info(f"Input tokens: {token_usage['inputTokens']}")
        logger.info(f"Output tokens: {token_usage['outputTokens']}")
        logger.info(f"Total tokens: {token_usage['totalTokens']}")
        logger.info(f"Stop reason: {response['stopReason']}")
        
        return response
    except ClientError as err:
        st.error(f"Error: {err.response['Error']['Message']}")
        logger.error(f"A client error occurred: {err.response['Error']['Message']}")
        return None

# ------- SAMPLE PROMPTS -------

ZERO_SHOT_PROMPTS = [
    {
        "name": "Math Problem - Basic",
        "prompt": "A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?",
        "cot_suffix": " Think step by step."
    },
    {
        "name": "Logic Puzzle - River Crossing",
        "prompt": "A farmer needs to cross a river with a fox, a chicken, and a sack of grain. The boat can only hold the farmer and one item. The fox can't be left alone with the chicken, and the chicken can't be left alone with the grain. How can the farmer get everything across?",
        "cot_suffix": " Think step by step."
    },
    {
        "name": "Science Question - Physics",
        "prompt": "Why does ice float in water? Explain the scientific principle behind this phenomenon.",
        "cot_suffix": " Think step by step."
    },
    {
        "name": "Ethical Dilemma - AI Monitoring",
        "prompt": "Is it ethical for a company to use AI to monitor its employees' productivity? Consider different stakeholder perspectives.",
        "cot_suffix": " Think step by step about different perspectives."
    },
    {
        "name": "Code Debugging - Python",
        "prompt": "What's wrong with this Python code and how would you fix it?\n\ndef calculate_average(numbers):\n    sum = 0\n    for num in numbers:\n        sum += num\n    return sum / len(numbers)\n\nresult = calculate_average([])",
        "cot_suffix": " Think step by step through the code execution."
    }
]

FEW_SHOT_PROMPTS = [
    {
        "name": "Math Word Problems",
        "prompt": """I'm going to solve some math word problems.

Example 1:
Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Answer: Roger starts with 5 tennis balls. He buys 2 cans, with 3 tennis balls each. So he gets 2 × 3 = 6 new tennis balls. Now he has 5 + 6 = 11 tennis balls.

Example 2:
Problem: The triangle has a base of 10 inches and a height of 8 inches. What is the area of the triangle?
Answer: The area of a triangle is (1/2) × base × height. So the area is (1/2) × 10 inches × 8 inches = 40 square inches.

Now solve this problem:
James bought 3 packages of chocolate bars. Each package has 6 bars. He ate 4 bars. How many bars does he have left?""",
        
        "cot_suffix": """

To solve this problem, I'll think step by step:

Example 1:
Problem: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Step 1: Roger starts with 5 tennis balls.
Step 2: He buys 2 cans of tennis balls, with 3 balls per can.
Step 3: The number of new balls is 2 × 3 = 6 tennis balls.
Step 4: The total number of tennis balls is 5 + 6 = 11 tennis balls.
Answer: 11 tennis balls

Example 2:
Problem: The triangle has a base of 10 inches and a height of 8 inches. What is the area of the triangle?
Step 1: The formula for the area of a triangle is (1/2) × base × height.
Step 2: Substitute the values: (1/2) × 10 inches × 8 inches
Step 3: Calculate: (1/2) × 80 square inches = 40 square inches
Answer: 40 square inches

Now for the new problem:
James bought 3 packages of chocolate bars. Each package has 6 bars. He ate 4 bars. How many bars does he have left?"""
    },
    {
        "name": "Logical Reasoning",
        "prompt": """I'm going to solve some logical reasoning problems.

Example 1:
Problem: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?
Answer: No, we cannot conclude that some roses fade quickly. While all roses are flowers, we only know that some flowers fade quickly. The flowers that fade quickly might not be roses.

Example 2:
Problem: If no humans can fly naturally, and all pilots are human, what can we conclude about pilots?
Answer: We can conclude that no pilots can fly naturally. Since all pilots are humans, and no humans can fly naturally, it follows that no pilots can fly naturally.

Now solve this problem:
If all doctors are busy people, and some busy people have stress, can we conclude that some doctors have stress?""",
        
        "cot_suffix": """

I'll solve these logical reasoning problems step by step:

Example 1:
Problem: If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly?
Step 1: We know that all roses are flowers (All A are B).
Step 2: We know that some flowers fade quickly (Some B are C).
Step 3: To conclude "some roses fade quickly" (Some A are C), we would need to know that the flowers that fade quickly include at least some roses.
Step 4: The information doesn't guarantee that the subset of flowers that fade quickly contains any roses.
Answer: No, we cannot conclude that some roses fade quickly.

Example 2:
Problem: If no humans can fly naturally, and all pilots are human, what can we conclude about pilots?
Step 1: We know that no humans can fly naturally (No A is B).
Step 2: We know that all pilots are humans (All C are A).
Step 3: Combining these statements: if all pilots are humans, and no humans can fly naturally, then no pilots can fly naturally.
Answer: No pilots can fly naturally.

Now for the new problem:
If all doctors are busy people, and some busy people have stress, can we conclude that some doctors have stress?"""
    },
    {
        "name": "Language Translation",
        "prompt": """I'll translate English phrases to French.

Example 1:
English: Hello, how are you today?
French: Bonjour, comment allez-vous aujourd'hui?

Example 2:
English: I would like to order a coffee, please.
French: Je voudrais commander un café, s'il vous plaît.

Now translate this phrase to French:
English: Where is the nearest train station?""",
        
        "cot_suffix": """

I'll translate these English phrases to French step by step:

Example 1:
English: Hello, how are you today?
Step 1: "Hello" in French is "Bonjour"
Step 2: "how are you" is formally translated as "comment allez-vous"
Step 3: "today" in French is "aujourd'hui"
Step 4: Putting it together with proper punctuation: "Bonjour, comment allez-vous aujourd'hui?"
French: Bonjour, comment allez-vous aujourd'hui?

Example 2:
English: I would like to order a coffee, please.
Step 1: "I would like" in French is "Je voudrais"
Step 2: "to order" translates to "commander"
Step 3: "a coffee" is "un café"
Step 4: "please" is "s'il vous plaît"
Step 5: Combining with proper grammar: "Je voudrais commander un café, s'il vous plaît."
French: Je voudrais commander un café, s'il vous plaît.

Now for the new phrase:
English: Where is the nearest train station?"""
    },
    {
        "name": "Medical Diagnosis",
        "prompt": """I'll analyze patient symptoms and provide possible diagnoses.

Example 1:
Patient: 45-year-old male with sudden chest pain radiating to left arm, shortness of breath, and sweating.
Diagnosis: These symptoms strongly suggest a myocardial infarction (heart attack). Immediate medical attention is required. The radiation of pain to the left arm is a classic sign of cardiac origin.

Example 2:
Patient: 8-year-old child with high fever, red rash that starts on face and spreads downward, cough, and red eyes.
Diagnosis: These symptoms are consistent with measles (rubeola). The characteristic rash pattern and associated symptoms suggest this viral infection, especially if the child hasn't been vaccinated.

Now provide a possible diagnosis:
Patient: 35-year-old female with severe headache, sensitivity to light, stiff neck, and fever.""",
        
        "cot_suffix": """

I'll analyze these patient symptoms step by step:

Example 1:
Patient: 45-year-old male with sudden chest pain radiating to left arm, shortness of breath, and sweating.
Step 1: Identify key symptoms: chest pain with left arm radiation, dyspnea (shortness of breath), diaphoresis (sweating).
Step 2: Consider common causes for this constellation of symptoms in this demographic.
Step 3: The combination of chest pain radiating to the left arm, along with shortness of breath and sweating, is highly specific for cardiac ischemia.
Step 4: Given the sudden onset in a middle-aged male, myocardial infarction is the most likely diagnosis.
Diagnosis: Myocardial infarction (heart attack)

Example 2:
Patient: 8-year-old child with high fever, red rash that starts on face and spreads downward, cough, and red eyes.
Step 1: Identify key symptoms: high fever, characteristic rash pattern (face to body), cough, conjunctivitis.
Step 2: Consider common childhood illnesses with these features.
Step 3: The pattern of rash beginning on the face and spreading downward is distinctive for measles.
Step 4: Supporting symptoms (cough, conjunctivitis, fever) further support this diagnosis.
Diagnosis: Measles (rubeola)

Now for the new patient:
Patient: 35-year-old female with severe headache, sensitivity to light, stiff neck, and fever."""
    },
    {
        "name": "Computer Science Algorithms",
        "prompt": """I'll explain algorithms and their time complexity.

Example 1:
Algorithm: Bubble Sort
Explanation: Bubble Sort works by repeatedly stepping through the list, comparing adjacent elements and swapping them if they are in the wrong order. The pass through the list is repeated until no swaps are needed. Its time complexity is O(n²) in the worst and average cases, making it inefficient for large lists.

Example 2:
Algorithm: Binary Search
Explanation: Binary Search operates on a sorted array by repeatedly dividing the search range in half. It compares the middle element with the target value. If they match, the position is returned. If the middle element is greater than the target, the search continues in the lower half; otherwise, the search continues in the upper half. Its time complexity is O(log n), making it very efficient for large datasets.

Now explain this algorithm:
Algorithm: Depth-First Search (DFS)""",
        
        "cot_suffix": """

I'll explain these algorithms step by step:

Example 1:
Algorithm: Bubble Sort
Step 1: Define what the algorithm does: Bubble Sort is a simple comparison-based sorting algorithm.
Step 2: Explain the mechanism: It works by repeatedly traversing the list, comparing adjacent elements and swapping them if they are in the wrong order.
Step 3: Describe the process: In each pass, the largest unsorted element "bubbles up" to its correct position.
Step 4: Analyze time complexity: Each pass requires n-1 comparisons, and we need up to n passes.
Step 5: Conclude with efficiency: This gives O(n²) time complexity in worst and average cases, making it inefficient for large datasets.
Explanation: Bubble Sort repeatedly compares adjacent elements and swaps them if needed, moving the largest elements to the end in each pass. Time complexity: O(n²).

Example 2:
Algorithm: Binary Search
Step 1: Define prerequisites: Binary Search requires a sorted array to function.
Step 2: Explain the core strategy: It uses a divide-and-conquer approach by repeatedly dividing the search space in half.
Step 3: Describe the process: Find the middle element, compare with target, and eliminate half the remaining elements.
Step 4: Analyze time complexity: Each step eliminates half the elements, leading to log₂(n) steps at most.
Step 5: Conclude with efficiency: This gives O(log n) time complexity, making it very efficient for large datasets.
Explanation: Binary Search works on sorted arrays by repeatedly dividing the search interval in half. Time complexity: O(log n).

Now for the new algorithm:
Algorithm: Depth-First Search (DFS)"""
    }
]

TOT_PROMPTS = [
    {
        "name": "ML Model Selection - Classification Task",
        "prompt": """You need to build a classification model to detect fraudulent credit card transactions. You have 100,000 transactions with 30 features, but only 0.5% are fraudulent (highly imbalanced). Which ML approach would you choose and why?""",
        
        "tot_suffix": """

I need to select the best ML approach for fraud detection with imbalanced data. Let me systematically explore different solution paths.

**Problem Analysis:**
- Dataset: 100,000 transactions, 30 features
- Class imbalance: 0.5% fraud (500 fraudulent, 99,500 legitimate)
- Goal: Detect fraudulent transactions with high recall

**Approach 1: Traditional ML with Resampling**
Path 1a: Random Forest + SMOTE
- Pros: Handles imbalance well, robust to outliers, feature importance
- Cons: May overfit on synthetic samples, slower training
- Expected performance: Precision ~70-80%, Recall ~75-85%
- Evaluation: Good baseline, interpretable ✓

Path 1b: XGBoost + Class Weights
- Pros: Excellent with imbalanced data, fast, handles missing values
- Cons: Requires careful hyperparameter tuning
- Expected performance: Precision ~75-85%, Recall ~80-90%
- Evaluation: Strong candidate, production-ready ✓✓

**Approach 2: Anomaly Detection Methods**
Path 2a: Isolation Forest
- Pros: Designed for anomaly detection, no need for balanced data
- Cons: Less interpretable, may miss subtle fraud patterns
- Expected performance: Precision ~60-70%, Recall ~70-80%
- Evaluation: Good for unsupervised scenarios

Path 2b: Autoencoder (Deep Learning)
- Pros: Learns complex patterns, good for rare events
- Cons: Requires more data, harder to tune, black box
- Expected performance: Precision ~65-75%, Recall ~75-85%
- Evaluation: Overkill for this dataset size

**Approach 3: Ensemble Strategy**
Path 3a: Stacking (XGBoost + Random Forest + Logistic Regression)
- Pros: Combines strengths of multiple models, highest accuracy potential
- Cons: Complex, slower inference, harder to maintain
- Expected performance: Precision ~80-90%, Recall ~85-92%
- Evaluation: Best performance but complex ✓✓✓

Path 3b: Voting Classifier (Soft voting)
- Pros: Simpler than stacking, reduces overfitting
- Cons: Still requires multiple models
- Expected performance: Precision ~75-85%, Recall ~82-88%
- Evaluation: Good balance of performance and complexity ✓✓

**Approach 4: Cost-Sensitive Learning**
Path 4a: Logistic Regression with Custom Loss
- Pros: Simple, interpretable, fast inference
- Cons: May not capture complex patterns
- Expected performance: Precision ~65-75%, Recall ~70-80%
- Evaluation: Good for baseline and explainability

**Evaluation Criteria:**
1. Performance (Recall priority): Approach 3a > 3b > 1b > 2b > 1a
2. Interpretability: 1a > 4a > 1b > 3b > 2a
3. Production readiness: 1b > 1a > 3b > 4a > 3a
4. Training time: 4a > 1a > 1b > 3b > 3a

**Recommended Solution:**
**Primary: XGBoost with Class Weights (Path 1b)**
- Use scale_pos_weight parameter to handle imbalance
- Apply stratified k-fold cross-validation
- Optimize for F2-score (emphasizes recall)
- Monitor precision-recall curve, not just accuracy
- Set decision threshold based on business cost of false negatives

**Alternative: Ensemble Approach (Path 3b) if higher performance needed**
- Combine XGBoost, Random Forest, and LightGBM
- Use soft voting with optimized weights
- Worth the complexity if fraud cost is very high

**Implementation Strategy:**
1. Start with XGBoost baseline
2. Tune hyperparameters (max_depth, learning_rate, scale_pos_weight)
3. If performance insufficient, move to ensemble
4. Always validate on time-based split (not random) for temporal patterns"""
    },
    {
        "name": "System Architecture - Microservices Design",
        "prompt": """You're designing a scalable e-commerce platform expected to handle 10,000 orders per hour during peak times. Should you use a monolithic architecture or microservices? Consider trade-offs and provide a recommendation.""",
        
        "tot_suffix": """

I need to choose between monolithic and microservices architecture for a high-traffic e-commerce platform. Let me explore multiple architectural paths systematically.

**Requirements Analysis:**
- Peak load: 10,000 orders/hour (~3 orders/second)
- Domain: E-commerce (catalog, cart, orders, payments, inventory, shipping)
- Scalability: Must handle traffic spikes
- Team: Assume medium-sized development team

**Approach 1: Pure Monolithic Architecture**
Path 1a: Traditional Monolith with Vertical Scaling
- Pros: Simple deployment, easier debugging, shared database, no network latency
- Cons: Single point of failure, hard to scale specific components, deployment risk
- Scalability: Limited by single server capacity
- Team impact: Easier for small teams, but merge conflicts increase
- Evaluation: Too risky for 10k orders/hour ✗

Path 1b: Modular Monolith with Horizontal Scaling
- Pros: Clear boundaries, easier to refactor later, load balancer distribution
- Cons: Still coupled deployment, database bottleneck
- Scalability: Can handle load with multiple instances + caching
- Team impact: Good for medium teams with clear module ownership
- Evaluation: Viable option, lower complexity ✓

**Approach 2: Full Microservices Architecture**
Path 2a: Fine-Grained Microservices (10+ services)
- Services: User, Product, Cart, Order, Payment, Inventory, Shipping, Notification, Analytics, Search
- Pros: Maximum flexibility, independent scaling, technology diversity
- Cons: High operational complexity, distributed transactions, network overhead
- Scalability: Excellent - scale each service independently
- Team impact: Requires DevOps expertise, service mesh, monitoring
- Evaluation: Over-engineered for this scale ✗

Path 2b: Coarse-Grained Microservices (4-5 services)
- Services: Catalog (products, search), Order Management (cart, orders), Payment, Fulfillment (inventory, shipping)
- Pros: Balanced complexity, key services can scale independently
- Cons: Some operational overhead, need API gateway, distributed tracing
- Scalability: Very good - scale order service during peaks
- Team impact: Manageable with 2-3 teams, clear service boundaries
- Evaluation: Strong candidate ✓✓

**Approach 3: Hybrid Architecture**
Path 3a: Monolith with Extracted Critical Services
- Core: Monolith for catalog, cart, user management
- Extracted: Payment service (PCI compliance), Order service (high load)
- Pros: Pragmatic, reduces complexity, isolates critical paths
- Cons: Some coupling remains, gradual migration needed
- Scalability: Good - scale order service independently
- Team impact: Easiest transition, learn microservices gradually
- Evaluation: Best balance for most teams ✓✓✓

Path 3b: Monolith with Event-Driven Extensions
- Core: Monolith for transactional operations
- Async: Event bus for notifications, analytics, recommendations
- Pros: Decouples non-critical paths, improves performance
- Cons: Eventual consistency challenges, message queue overhead
- Scalability: Good for async workloads
- Team impact: Moderate complexity, need message broker expertise
- Evaluation: Good for specific use cases ✓✓

**Approach 4: Serverless Architecture**
Path 4a: AWS Lambda + API Gateway
- Pros: Auto-scaling, pay-per-use, no server management
- Cons: Cold starts, vendor lock-in, debugging complexity, cost at scale
- Scalability: Excellent auto-scaling
- Team impact: Different skillset required
- Evaluation: Interesting but risky for core e-commerce ✗

**Decision Matrix:**

| Criteria | Monolith (1b) | Microservices (2b) | Hybrid (3a) |
|----------|---------------|-------------------|-------------|
| Scalability | 7/10 | 10/10 | 9/10 |
| Complexity | 9/10 | 4/10 | 7/10 |
| Team Readiness | 9/10 | 5/10 | 8/10 |
| Time to Market | 10/10 | 5/10 | 8/10 |
| Operational Cost | 8/10 | 5/10 | 7/10 |
| **Total** | **43/50** | **29/50** | **39/50** |

**Recommended Solution: Hybrid Architecture (Path 3a)**

**Phase 1: Start with Modular Monolith**
- Build with clear module boundaries
- Use domain-driven design principles
- Implement async communication where possible
- Set up comprehensive monitoring

**Phase 2: Extract Critical Services**
- Extract Payment Service first (compliance, security)
- Extract Order Service second (high load, business critical)
- Keep Catalog, Cart, User in monolith (lower traffic, tightly coupled)

**Phase 3: Scale Strategically**
- Horizontal scaling for monolith (3-5 instances)
- Independent scaling for Order Service (10+ instances during peaks)
- Use Redis for session management and caching
- Implement CDN for static assets

**Why This Works:**
- Handles 10k orders/hour easily (Order service scales independently)
- Lower operational complexity than full microservices
- Faster time to market than building microservices from scratch
- Clear migration path if you need more services later
- Team can learn microservices patterns gradually

**Technology Stack:**
- Monolith: Spring Boot or Django
- Order Service: Node.js or Go (high concurrency)
- Payment Service: Java (security, compliance)
- Message Queue: RabbitMQ or AWS SQS
- Cache: Redis
- Database: PostgreSQL (main), MongoDB (product catalog)
- API Gateway: Kong or AWS API Gateway"""
    },
    {
        "name": "Prompt Engineering - Hallucination Mitigation",
        "prompt": """An LLM keeps generating factually incorrect information about recent events. What strategies would you use to reduce hallucinations and improve factual accuracy?""",
        
        "tot_suffix": """

I need to reduce LLM hallucinations and improve factual accuracy. Let me systematically explore different mitigation strategies.

**Problem Analysis:**
- Issue: LLM generating false information about recent events
- Root causes: Training data cutoff, lack of grounding, overconfidence
- Goal: Improve factual accuracy while maintaining usefulness

**Approach 1: Retrieval-Augmented Generation (RAG)**
Path 1a: Basic RAG with Vector Database
- Implementation: Embed queries, retrieve relevant docs, inject into context
- Pros: Grounds responses in real data, handles recent information
- Cons: Retrieval quality critical, context window limits, latency increase
- Expected improvement: 60-80% reduction in hallucinations
- Evaluation: Strong baseline solution ✓✓

Path 1b: Advanced RAG with Reranking
- Implementation: Multi-stage retrieval → rerank → generate with citations
- Pros: Better relevance, source attribution, verifiable claims
- Cons: More complex, higher latency, requires reranking model
- Expected improvement: 70-85% reduction in hallucinations
- Evaluation: Best for high-stakes applications ✓✓✓

Path 1c: Hybrid Search (Dense + Sparse)
- Implementation: Combine vector search with keyword search (BM25)
- Pros: Better recall, handles both semantic and exact matches
- Cons: More infrastructure, tuning required
- Expected improvement: 65-80% reduction
- Evaluation: Good for diverse query types ✓✓

**Approach 2: Prompt Engineering Techniques**
Path 2a: Explicit Uncertainty Instructions
- Prompt: "If you're not certain, say 'I don't know' rather than guessing"
- Pros: Simple, no infrastructure, reduces overconfidence
- Cons: May refuse too often, doesn't add new knowledge
- Expected improvement: 30-40% reduction
- Evaluation: Easy first step ✓

Path 2b: Chain-of-Thought with Verification
- Prompt: "Think step-by-step and verify each claim before stating it"
- Pros: Encourages self-reflection, better reasoning
- Cons: Longer responses, still limited by training data
- Expected improvement: 35-45% reduction
- Evaluation: Good for reasoning tasks ✓

Path 2c: Few-Shot with Correct Examples
- Prompt: Provide examples of accurate, well-sourced responses
- Pros: Demonstrates desired behavior, improves consistency
- Cons: Uses context window, examples must be carefully chosen
- Expected improvement: 40-50% reduction
- Evaluation: Effective for specific domains ✓✓

**Approach 3: Model Fine-Tuning**
Path 3a: Fine-Tune on Factual QA Dataset
- Implementation: RLHF or supervised fine-tuning on verified facts
- Pros: Permanent improvement, no runtime overhead
- Cons: Expensive, requires large dataset, may reduce generality
- Expected improvement: 50-70% reduction
- Evaluation: Good for specific domains with budget ✓✓

Path 3b: Instruction Tuning for Uncertainty
- Implementation: Train model to express uncertainty appropriately
- Pros: Better calibration, more honest responses
- Cons: Requires careful dataset curation
- Expected improvement: 45-60% reduction
- Evaluation: Improves trustworthiness ✓✓

**Approach 4: Multi-Model Verification**
Path 4a: Ensemble with Fact-Checking Model
- Implementation: Generate response → fact-check with specialized model → revise
- Pros: Catches errors before user sees them, high accuracy
- Cons: 2-3x latency, higher cost, complex pipeline
- Expected improvement: 75-90% reduction
- Evaluation: Best accuracy but expensive ✓✓✓

Path 4b: Self-Consistency Checking
- Implementation: Generate multiple responses, check for consistency
- Pros: No additional models needed, catches contradictions
- Cons: Higher latency and cost (multiple generations)
- Expected improvement: 50-65% reduction
- Evaluation: Good middle ground ✓✓

**Approach 5: Structured Output with Constraints**
Path 5a: JSON Schema with Required Fields
- Implementation: Force structured output with source citations
- Pros: Verifiable, parseable, encourages grounding
- Cons: Less natural, may not fit all use cases
- Expected improvement: 55-70% reduction
- Evaluation: Excellent for data extraction ✓✓

Path 5b: Template-Based Generation
- Implementation: Fill predefined templates with verified information
- Pros: Highly controlled, consistent format
- Cons: Less flexible, may feel robotic
- Expected improvement: 60-75% reduction
- Evaluation: Good for repetitive tasks ✓

**Decision Matrix:**

| Strategy | Accuracy | Cost | Latency | Complexity | Score |
|----------|----------|------|---------|------------|-------|
| Advanced RAG (1b) | 9/10 | 6/10 | 6/10 | 7/10 | 28/40 |
| Basic RAG (1a) | 8/10 | 7/10 | 7/10 | 8/10 | 30/40 |
| Multi-Model (4a) | 10/10 | 4/10 | 4/10 | 5/10 | 23/40 |
| Fine-Tuning (3a) | 7/10 | 5/10 | 9/10 | 6/10 | 27/40 |
| Prompt Eng (2c) | 6/10 | 9/10 | 9/10 | 9/10 | 33/40 |

**Recommended Solution: Layered Defense Strategy**

**Tier 1: Immediate Improvements (Week 1)**
1. Update system prompt with uncertainty instructions
2. Add few-shot examples of well-sourced responses
3. Implement basic input validation

**Tier 2: RAG Implementation (Weeks 2-4)**
1. Set up vector database (Pinecone, Weaviate, or ChromaDB)
2. Implement basic RAG pipeline:
   - Query → Embed → Retrieve top-k docs → Inject context → Generate
3. Add source citations to responses
4. Monitor retrieval quality metrics

**Tier 3: Advanced Techniques (Months 2-3)**
1. Implement reranking (Cohere Rerank or cross-encoder)
2. Add hybrid search (vector + BM25)
3. Implement confidence scoring
4. A/B test different retrieval strategies

**Tier 4: Verification Layer (Month 4+)**
1. Add fact-checking model for high-stakes queries
2. Implement self-consistency checking for critical facts
3. Build feedback loop to improve retrieval

**Implementation Example:**

```python
# System prompt with uncertainty handling
system_prompt = '''You are a helpful assistant. Follow these rules:
1. Only state facts you're confident about
2. Cite sources when available: [Source: ...]
3. If uncertain, say "I don't have reliable information about..."
4. For recent events, acknowledge your knowledge cutoff
5. Never make up dates, statistics, or quotes'''

# RAG-enhanced generation
def generate_with_rag(query):
    # Retrieve relevant documents
    docs = vector_db.search(query, top_k=5)
    
    # Build context
    context = "\\n\\n".join([f"[Source {i+1}]: {doc}" 
                            for i, doc in enumerate(docs)])
    
    # Enhanced prompt
    prompt = f'''Context information:
{context}

Question: {query}

Instructions: Answer based on the context above. Cite sources using [Source N].
If the context doesn't contain the answer, say so clearly.'''
    
    return llm.generate(prompt, system=system_prompt)
```

**Expected Results:**
- Week 1: 30-40% reduction in hallucinations
- Month 1: 60-75% reduction with RAG
- Month 3: 75-85% reduction with advanced techniques
- Month 4+: 85-90% reduction with verification layer

**Monitoring Metrics:**
- Factual accuracy rate (human eval)
- Citation rate (% responses with sources)
- Uncertainty expression rate
- User satisfaction scores
- Retrieval relevance (precision@k)"""
    },
    {
        "name": "Data Pipeline Optimization - ETL Performance",
        "prompt": """Your ETL pipeline processes 500GB of data daily but takes 8 hours to complete. The business needs it done in 2 hours. What optimization strategies would you explore?""",
        
        "tot_suffix": """

I need to reduce ETL pipeline time from 8 hours to 2 hours (4x speedup). Let me systematically explore optimization strategies.

**Current State Analysis:**
- Data volume: 500GB daily
- Current time: 8 hours (62.5 GB/hour)
- Target time: 2 hours (250 GB/hour)
- Required speedup: 4x

**Approach 1: Parallelization Strategies**
Path 1a: Horizontal Partitioning (Data Parallelism)
- Implementation: Split data into chunks, process in parallel
- Example: 10 workers processing 50GB each
- Pros: Linear scalability, simple to implement
- Cons: Requires partitionable data, coordination overhead
- Expected speedup: 3-5x (depending on workers)
- Cost: Medium (more compute resources)
- Evaluation: Strong candidate ✓✓✓

Path 1b: Pipeline Parallelism (Stage Parallelism)
- Implementation: Overlap Extract, Transform, Load stages
- Example: While batch N loads, batch N+1 transforms, batch N+2 extracts
- Pros: Better resource utilization, no data splitting needed
- Cons: Complex coordination, requires streaming architecture
- Expected speedup: 2-3x
- Cost: Low (same resources, better utilization)
- Evaluation: Good complementary approach ✓✓

Path 1c: Hybrid Parallelism
- Implementation: Combine data + pipeline parallelism
- Pros: Maximum throughput, flexible scaling
- Cons: Most complex, harder to debug
- Expected speedup: 5-8x
- Cost: High (complexity + resources)
- Evaluation: Overkill unless needed for future growth ✓

**Approach 2: Infrastructure Optimization**
Path 2a: Upgrade to Distributed Processing (Spark/Dask)
- Implementation: Replace single-node with Spark cluster
- Pros: Built-in parallelism, fault tolerance, mature ecosystem
- Cons: Learning curve, infrastructure overhead, cost
- Expected speedup: 4-6x
- Cost: High (cluster + maintenance)
- Evaluation: Best for long-term scalability ✓✓✓

Path 2b: Use Columnar Storage (Parquet/ORC)
- Implementation: Convert source data to columnar format
- Pros: Faster reads (only needed columns), better compression
- Cons: Requires data conversion, storage format change
- Expected speedup: 2-3x (for read-heavy workloads)
- Cost: Low (storage optimization)
- Evaluation: Easy win for analytics workloads ✓✓

Path 2c: Increase Compute Resources (Vertical Scaling)
- Implementation: Bigger machine (more CPU, RAM, faster disk)
- Pros: Simple, no code changes
- Cons: Limited scalability, expensive, diminishing returns
- Expected speedup: 1.5-2x
- Cost: High (hardware costs)
- Evaluation: Quick fix but not sustainable ✗

**Approach 3: Algorithm & Code Optimization**
Path 3a: Optimize Transformations
- Implementation: Profile code, optimize hot paths, vectorize operations
- Examples: Use pandas vectorization, avoid loops, optimize SQL queries
- Pros: No infrastructure changes, permanent improvement
- Cons: Requires deep analysis, may be limited gains
- Expected speedup: 1.5-3x
- Cost: Low (developer time)
- Evaluation: Should always do this first ✓✓✓

Path 3b: Reduce Data Movement
- Implementation: Push transformations to source (SQL), avoid unnecessary copies
- Examples: Filter early, aggregate at source, use views
- Pros: Less network I/O, less memory usage
- Cons: May shift load to source database
- Expected speedup: 1.5-2.5x
- Cost: Very low
- Evaluation: Quick wins available ✓✓

Path 3c: Incremental Processing
- Implementation: Process only changed data (CDC - Change Data Capture)
- Pros: Dramatically reduces data volume, faster processing
- Cons: Requires change tracking, more complex logic
- Expected speedup: 5-10x (if only 10-20% data changes daily)
- Cost: Medium (implementation complexity)
- Evaluation: Game-changer if applicable ✓✓✓

**Approach 4: Caching & Precomputation**
Path 4a: Materialize Intermediate Results
- Implementation: Cache expensive transformations, reuse across runs
- Pros: Avoid recomputation, faster iterations
- Cons: Storage overhead, cache invalidation complexity
- Expected speedup: 2-4x (for repeated computations)
- Cost: Low (storage)
- Evaluation: Good for complex pipelines ✓✓

Path 4b: Precompute Aggregations
- Implementation: Maintain summary tables, update incrementally
- Pros: Much faster queries, reduced processing
- Cons: More storage, consistency challenges
- Expected speedup: 3-5x (for aggregation-heavy workloads)
- Cost: Low-Medium
- Evaluation: Excellent for reporting pipelines ✓✓

**Approach 5: Architectural Changes**
Path 5a: Streaming Architecture (Kafka + Flink)
- Implementation: Replace batch with real-time streaming
- Pros: Continuous processing, lower latency, better resource utilization
- Cons: Major rewrite, operational complexity, different paradigm
- Expected speedup: N/A (different model, but spreads load over 24 hours)
- Cost: Very high (rewrite + infrastructure)
- Evaluation: Future-proof but major undertaking ✓

Path 5b: Lambda Architecture (Batch + Stream)
- Implementation: Fast path for recent data, batch for historical
- Pros: Best of both worlds, handles late data
- Cons: Maintain two systems, complexity
- Expected speedup: Varies
- Cost: Very high
- Evaluation: Only for specific requirements ✗

**Decision Matrix:**

| Strategy | Speedup | Cost | Complexity | Time to Implement | Score |
|----------|---------|------|------------|-------------------|-------|
| Incremental (3c) | 10/10 | 7/10 | 6/10 | 7/10 | 30/40 |
| Spark Cluster (2a) | 9/10 | 5/10 | 6/10 | 5/10 | 25/40 |
| Data Parallel (1a) | 8/10 | 7/10 | 8/10 | 8/10 | 31/40 |
| Code Optimize (3a) | 7/10 | 9/10 | 8/10 | 9/10 | 33/40 |
| Columnar (2b) | 7/10 | 9/10 | 9/10 | 8/10 | 33/40 |

**Recommended Solution: Phased Optimization**

**Phase 1: Quick Wins (Week 1-2) - Target: 2x speedup**
1. Profile current pipeline to identify bottlenecks
2. Optimize hot paths:
   - Vectorize pandas operations
   - Optimize SQL queries (add indexes, push filters down)
   - Remove unnecessary data copies
3. Convert to columnar storage (Parquet)
4. Implement basic caching for expensive operations

**Phase 2: Parallelization (Week 3-4) - Target: 2x additional speedup**
1. Implement data partitioning:
   - Partition by date, customer_id, or natural key
   - Process partitions in parallel (multiprocessing or ThreadPoolExecutor)
2. Add pipeline parallelism:
   - Use queues to overlap Extract/Transform/Load
3. Optimize I/O:
   - Batch database operations
   - Use connection pooling
   - Parallel reads/writes

**Phase 3: Incremental Processing (Week 5-6) - Target: Maintain performance**
1. Implement Change Data Capture:
   - Track changed records (timestamps, version columns)
   - Process only deltas
2. Maintain state:
   - Store last processed timestamp
   - Handle late-arriving data

**Implementation Example:**

```python
# Phase 1: Optimized transformation
def transform_optimized(df):
    # Vectorized operations instead of loops
    df['total'] = df['quantity'] * df['price']  # Not: df.apply()
    
    # Filter early
    df = df[df['status'] == 'active']
    
    # Use categorical for low-cardinality columns
    df['category'] = df['category'].astype('category')
    
    return df

# Phase 2: Parallel processing
from multiprocessing import Pool
from functools import partial

def process_partition(partition_id, date):
    # Each worker processes one partition
    df = extract_partition(partition_id, date)
    df = transform_optimized(df)
    load_partition(df, partition_id)

def run_parallel_etl(date, num_workers=10):
    partitions = range(num_workers)
    with Pool(num_workers) as pool:
        pool.map(partial(process_partition, date=date), partitions)

# Phase 3: Incremental processing
def run_incremental_etl(date):
    last_processed = get_last_processed_timestamp()
    
    # Only extract changed records
    df = extract_changes_since(last_processed)
    
    if df.empty:
        print("No changes to process")
        return
    
    df = transform_optimized(df)
    load_incremental(df)  # Upsert instead of full load
    
    update_last_processed_timestamp(date)
```

**Expected Results:**
- Week 2: 4 hours (2x speedup from optimization + columnar storage)
- Week 4: 2 hours (4x total speedup with parallelization)
- Week 6: 1-2 hours (4-8x speedup with incremental processing)

**Monitoring & Validation:**
- Add timing metrics for each stage
- Monitor resource utilization (CPU, memory, I/O)
- Track data quality metrics
- Set up alerts for pipeline failures
- A/B test optimizations before full rollout"""
    },
    {
        "name": "Security Incident Response - API Key Leak",
        "prompt": """Your team accidentally committed AWS API keys to a public GitHub repository. The commit was public for 3 hours before being noticed. What immediate actions should you take?""",
        
        "tot_suffix": """

I need to respond to an AWS API key leak in a public GitHub repository. Let me systematically explore the incident response paths.

**Incident Assessment:**
- Exposure: AWS API keys in public GitHub repo
- Duration: 3 hours public exposure
- Risk: High - automated scanners find keys within minutes
- Assumption: Keys may already be compromised

**Approach 1: Immediate Containment (First 15 minutes)**
Path 1a: Revoke Compromised Credentials
- Action 1: Log into AWS Console immediately
- Action 2: Navigate to IAM → Users → Security Credentials
- Action 3: Delete the exposed access keys (make inactive first if unsure)
- Action 4: Generate new keys and securely distribute to team
- Pros: Stops ongoing unauthorized access immediately
- Cons: May break running services temporarily
- Priority: CRITICAL - Do this first ✓✓✓

Path 1b: Enable MFA Delete (if not already enabled)
- Action: Enable MFA for critical operations
- Pros: Adds protection layer
- Cons: Doesn't help with already-leaked keys
- Priority: Important but secondary

Path 1c: Rotate All Related Credentials
- Action: Rotate database passwords, other API keys that might be exposed
- Pros: Comprehensive security
- Cons: Time-consuming, may cause outages
- Priority: Do after immediate containment ✓✓

**Approach 2: Assess Damage (First 30 minutes)**
Path 2a: Check CloudTrail Logs
- Action: Review CloudTrail for unauthorized API calls
- Look for:
  - Unusual regions (especially crypto-mining regions)
  - EC2 instance launches (especially GPU instances)
  - S3 bucket access or data exfiltration
  - IAM policy changes
  - Resource deletions
- Pros: Identifies what attacker did
- Cons: Logs may have delay (up to 15 min)
- Priority: CRITICAL - Do immediately after revocation ✓✓✓

Path 2b: Check AWS Cost Explorer
- Action: Look for unexpected charges
- Common attack patterns:
  - EC2 instances for crypto mining
  - Data transfer charges (exfiltration)
  - Lambda invocations
- Pros: Quick indicator of resource abuse
- Cons: Charges may not appear immediately
- Priority: Important for financial impact ✓✓

Path 2c: Review GuardDuty Findings (if enabled)
- Action: Check GuardDuty for security alerts
- Pros: Automated threat detection
- Cons: Only if previously enabled
- Priority: Check if available ✓

**Approach 3: Remove Evidence from GitHub (First 30 minutes)**
Path 3a: Delete Commit from History (git filter-branch)
- Action: Use git filter-branch or BFG Repo-Cleaner to remove keys
- Commands:
  ```
  git filter-branch --force --index-filter \
    "git rm --cached --ignore-unmatch path/to/file" \
    --prune-empty --tag-name-filter cat -- --all
  git push origin --force --all
  ```
- Pros: Removes keys from all history
- Cons: Doesn't help if already scraped, breaks forks
- Priority: Important but keys already compromised ✓✓

Path 3b: Make Repository Private Temporarily
- Action: Change repo visibility to private
- Pros: Quick, prevents further exposure
- Cons: Doesn't remove from caches/scrapers
- Priority: Quick action while cleaning history ✓

Path 3c: Contact GitHub Support
- Action: Request GitHub to purge cached versions
- Pros: Removes from GitHub's caches
- Cons: Slow, keys likely already scraped
- Priority: Do it but don't rely on it ✓

**Approach 4: Prevent Future Unauthorized Access (First hour)**
Path 4a: Implement SCPs (Service Control Policies)
- Action: Restrict which services can be used
- Example: Deny EC2 in expensive regions, deny GPU instances
- Pros: Limits blast radius of future compromises
- Cons: May restrict legitimate use
- Priority: Important for prevention ✓✓

Path 4b: Set Up Billing Alerts
- Action: Create CloudWatch alarms for unusual spending
- Thresholds: Alert if daily spend > 2x normal
- Pros: Early warning system
- Cons: Reactive, not preventive
- Priority: Should have been done already ✓✓

Path 4c: Enable AWS Config Rules
- Action: Set up compliance rules for security best practices
- Pros: Continuous monitoring
- Cons: Doesn't prevent initial compromise
- Priority: Good for long-term ✓

**Approach 5: Communication & Documentation (First 2 hours)**
Path 5a: Notify Stakeholders
- Internal: Security team, management, affected service owners
- External: Customers (if their data was accessed)
- Pros: Transparency, compliance
- Cons: May cause panic if not handled well
- Priority: Critical for compliance ✓✓✓

Path 5b: Document Incident
- Action: Create incident report with timeline
- Include: What happened, when, impact, actions taken
- Pros: Required for compliance, helps prevent recurrence
- Cons: Time-consuming during crisis
- Priority: Do concurrently with response ✓✓

Path 5c: File Incident Report with AWS
- Action: Contact AWS Support about potential compromise
- Pros: AWS may provide additional insights or assistance
- Cons: May trigger account review
- Priority: Important for serious incidents ✓✓

**Approach 6: Long-Term Prevention (Days 1-7)**
Path 6a: Implement Secrets Management
- Action: Use AWS Secrets Manager or HashiCorp Vault
- Pros: Centralized, rotatable, auditable
- Cons: Requires code changes
- Priority: Essential to prevent recurrence ✓✓✓

Path 6b: Set Up Pre-Commit Hooks
- Action: Use tools like git-secrets, truffleHog
- Pros: Prevents commits with secrets
- Cons: Can be bypassed, may have false positives
- Priority: Easy win for prevention ✓✓✓

Path 6c: Implement Least Privilege IAM
- Action: Review and restrict IAM permissions
- Principle: Only grant minimum necessary permissions
- Pros: Limits damage from future compromises
- Cons: Requires careful planning
- Priority: Critical security practice ✓✓✓

**Recommended Action Plan:**

**Immediate (0-15 minutes):**
1. ✅ Revoke exposed AWS access keys in IAM
2. ✅ Generate new keys, distribute securely
3. ✅ Make GitHub repo private

**Short-term (15-60 minutes):**
4. ✅ Review CloudTrail logs for unauthorized activity
5. ✅ Check AWS Cost Explorer for unexpected charges
6. ✅ Terminate any unauthorized resources (EC2, Lambda, etc.)
7. ✅ Remove keys from git history (git filter-branch)
8. ✅ Enable GuardDuty if not already enabled
9. ✅ Set up billing alerts

**Medium-term (1-4 hours):**
10. ✅ Notify security team and management
11. ✅ Document incident timeline and actions
12. ✅ Contact AWS Support if significant unauthorized activity found
13. ✅ Review and restrict IAM permissions
14. ✅ Implement SCPs to limit blast radius

**Long-term (1-7 days):**
15. ✅ Implement AWS Secrets Manager
16. ✅ Set up pre-commit hooks (git-secrets)
17. ✅ Conduct security training for team
18. ✅ Implement automated secret scanning in CI/CD
19. ✅ Review and update incident response procedures
20. ✅ Conduct post-mortem meeting

**Critical Commands:**

```bash
# 1. Revoke keys (AWS CLI)
aws iam delete-access-key --access-key-id AKIAIOSFODNN7EXAMPLE --user-name username

# 2. Check CloudTrail for unauthorized activity
aws cloudtrail lookup-events --lookup-attributes AttributeKey=Username,AttributeValue=username --max-results 50

# 3. List all EC2 instances (check for unauthorized)
aws ec2 describe-instances --query 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType,LaunchTime]' --output table

# 4. Remove secrets from git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch config/credentials.yml" \
  --prune-empty --tag-name-filter cat -- --all

# 5. Install git-secrets
git secrets --install
git secrets --register-aws
```

**Expected Outcomes:**
- Immediate: Unauthorized access stopped
- 1 hour: Full damage assessment complete
- 4 hours: All unauthorized resources terminated
- 1 day: Preventive measures in place
- 1 week: Comprehensive security improvements deployed"""
    }
]

# ------- SESSION STATE INITIALIZATION -------

def init_session_state():
    """Initialize session state variables."""
        
    # Zero-shot state variables
    if 'zero_shot_prompt' not in st.session_state:
        st.session_state.zero_shot_prompt = ZERO_SHOT_PROMPTS[0]["prompt"]
    if 'zero_shot_system_prompt' not in st.session_state:
        st.session_state.zero_shot_system_prompt = "You are a helpful and knowledgeable assistant who provides accurate information."
    if 'zero_shot_standard_response' not in st.session_state:
        st.session_state.zero_shot_standard_response = None
    if 'zero_shot_cot_response' not in st.session_state:
        st.session_state.zero_shot_cot_response = None
    
    # Few-shot state variables
    if 'few_shot_prompt' not in st.session_state:
        st.session_state.few_shot_prompt = FEW_SHOT_PROMPTS[0]["prompt"]
    if 'few_shot_cot_prompt' not in st.session_state:
        st.session_state.few_shot_cot_prompt = FEW_SHOT_PROMPTS[0]["prompt"] + FEW_SHOT_PROMPTS[0]["cot_suffix"]
    if 'few_shot_system_prompt' not in st.session_state:
        st.session_state.few_shot_system_prompt = "You are a helpful and knowledgeable assistant who provides accurate information."
    if 'few_shot_standard_response' not in st.session_state:
        st.session_state.few_shot_standard_response = None
    if 'few_shot_cot_response' not in st.session_state:
        st.session_state.few_shot_cot_response = None
    
    # Tree of Thoughts state variables
    if 'tot_prompt' not in st.session_state:
        st.session_state.tot_prompt = TOT_PROMPTS[0]["prompt"]
    if 'tot_cot_prompt' not in st.session_state:
        st.session_state.tot_cot_prompt = TOT_PROMPTS[0]["prompt"] + TOT_PROMPTS[0]["tot_suffix"]
    if 'tot_system_prompt' not in st.session_state:
        st.session_state.tot_system_prompt = "You are a helpful and knowledgeable assistant who provides accurate information and can explore multiple solution paths systematically."
    if 'tot_standard_response' not in st.session_state:
        st.session_state.tot_standard_response = None
    if 'tot_tot_response' not in st.session_state:
        st.session_state.tot_tot_response = None
    
    # Common state variables
    if 'analysis_shown' not in st.session_state:
        st.session_state.analysis_shown = False
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Zero-Shot"

# ------- UI COMPONENTS -------

def parameter_sidebar():
    """Sidebar with model selection and parameter tuning."""
    
    with st.container(border=True):
    
        st.markdown("<div class='sub-header'>Model Selection</div>", unsafe_allow_html=True)
        
        MODEL_CATEGORIES = {
        "Amazon": ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-pro-v1:0", 
                  "us.amazon.nova-2-lite-v1:0"],
        "Anthropic": ["anthropic.claude-3-haiku-20240307-v1:0",
                         "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                         "us.anthropic.claude-sonnet-4-20250514-v1:0",
                         "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                         "us.anthropic.claude-opus-4-1-20250805-v1:0"],
        "Cohere": ["cohere.command-r-v1:0", "cohere.command-r-plus-v1:0"],
        "Google": ["google.gemma-3-4b-it", "google.gemma-3-12b-it", "google.gemma-3-27b-it"],
        "Meta": ["us.meta.llama3-2-1b-instruct-v1:0", "us.meta.llama3-2-3b-instruct-v1:0",
                    "meta.llama3-8b-instruct-v1:0", "us.meta.llama3-1-8b-instruct-v1:0",
                    "us.meta.llama4-scout-17b-instruct-v1:0", "us.meta.llama4-maverick-17b-instruct-v1:0",
                    "meta.llama3-70b-instruct-v1:0", "us.meta.llama3-1-70b-instruct-v1:0",
                    "us.meta.llama3-3-70b-instruct-v1:0",
                    "us.meta.llama3-2-11b-instruct-v1:0", "us.meta.llama3-2-90b-instruct-v1:0"],
        "Mistral": ["mistral.mistral-7b-instruct-v0:2", "mistral.mistral-small-2402-v1:0",
                       "mistral.mistral-large-2402-v1:0", "mistral.mixtral-8x7b-instruct-v0:1"],
        "NVIDIA": ["nvidia.nemotron-nano-9b-v2", "nvidia.nemotron-nano-12b-v2"],
        "OpenAI": ["openai.gpt-oss-20b-1:0", "openai.gpt-oss-120b-1:0"],
        "Qwen": ["qwen.qwen3-32b-v1:0", "qwen.qwen3-next-80b-a3b", "qwen.qwen3-235b-a22b-2507-v1:0", "qwen.qwen3-vl-235b-a22b", "qwen.qwen3-coder-30b-a3b-v1:0", "qwen.qwen3-coder-480b-a35b-v1:0"],
        "Writer": ["us.writer.palmyra-x4-v1:0", "us.writer.palmyra-x5-v1:0"]
        }
        
        # Create selectbox for provider first
        provider = st.selectbox("Select Provider", options=list(MODEL_CATEGORIES.keys()))
        
        # Then create selectbox for models from that provider
        model_id = st.selectbox("Select Model", options=MODEL_CATEGORIES[provider])
        
        st.markdown("<div class='sub-header'>Parameter Tuning</div>", unsafe_allow_html=True)
        
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1, 
                            help="Higher values make output more random, lower values more deterministic")
        
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.9, step=0.1,
                            help="Controls diversity via nucleus sampling")
        
        max_tokens = st.number_input("Max Tokens", min_value=50, max_value=4096, value=1024, step=50,
                                    help="Maximum number of tokens in the response")
    with st.sidebar:
        common.render_sidebar()
        
        with st.expander("About Prompting Techniques", expanded=False):
            st.markdown("""
            ### Chain-of-Thought (CoT) & Tree of Thoughts (ToT)
            
            Advanced prompting techniques that enhance reasoning in AI models:
            
            **Chain-of-Thought (CoT):**
            - Encourages step-by-step reasoning
            - Improves multi-step problem solving
            - Two types: Zero-Shot and Few-Shot
            
            **Tree of Thoughts (ToT):**
            - Explores multiple reasoning paths
            - Systematically evaluates different approaches
            - Combines and compares solution strategies
            - Best for complex problems with multiple solutions
            
            **Key Benefits:**
            - Enhanced reasoning capabilities
            - Better problem decomposition
            - More thorough analysis
            - Higher accuracy on complex tasks
            """)
        
        params = {
            "temperature": temperature,
            "topP": top_p,
            "maxTokens": max_tokens
        }
        
    return model_id, params

def display_sample_prompts(prompts, current_prompt, key_prefix, on_change_callback=None):
    """Display sample prompts as a selectbox."""
    prompt_names = [p["name"] for p in prompts]
    
    # Find the current index
    current_index = 0
    for i, p in enumerate(prompts):
        if p["prompt"] == current_prompt:
            current_index = i
            break
    
    # Create the selectbox with callback
    selected_name = st.selectbox(
        "Select a sample prompt:", 
        options=prompt_names, 
        key=f"{key_prefix}_select",
        index=current_index,
        on_change=on_change_callback
    )
    
    # Return the selected prompt data
    selected_prompt = next(p for p in prompts if p["name"] == selected_name)
    return selected_prompt

def display_responses(standard_response, enhanced_response, enhanced_type="CoT"):
    """Display the standard and enhanced responses side by side."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='comparison-header'>Standard Prompting</div>", unsafe_allow_html=True)
        
        if standard_response:
            output_message = standard_response['output']['message']
            for content in output_message['content']:
                if 'text' in content:
                    st.markdown(content['text'])
                
            token_usage = standard_response['usage']
            st.caption(f"Input: {token_usage['inputTokens']} | Output: {token_usage['outputTokens']} | Total: {token_usage['totalTokens']}")
        else:
            st.caption("Response will appear here...")
            
    with col2:
        header_text = "Tree of Thoughts Prompting" if enhanced_type == "ToT" else "Chain-of-Thought Prompting"
        st.markdown(f"<div class='comparison-header'>{header_text}</div>", unsafe_allow_html=True)
        
        if enhanced_response:
            output_message = enhanced_response['output']['message']
            for content in output_message['content']:
                if 'text' in content:
                    st.markdown(content['text'])
                
            token_usage = enhanced_response['usage']
            st.caption(f"Input: {token_usage['inputTokens']} | Output: {token_usage['outputTokens']} | Total: {token_usage['totalTokens']}")
        else:
            st.caption("Response will appear here...")

def display_analysis(standard_response, enhanced_response, user_prompt, analysis_shown, enhanced_type="CoT"):
    """Display analysis of the differences between standard and enhanced responses."""
    if standard_response and enhanced_response and analysis_shown:
        st.markdown("### Response Analysis")
        st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
        
        # Extract responses
        standard_text = standard_response['output']['message']['content'][0].get('text', '')
        enhanced_text = enhanced_response['output']['message']['content'][0].get('text', '')
        
        # Get token usage metrics
        standard_tokens = standard_response['usage']
        enhanced_tokens = enhanced_response['usage']
        
        # Create a prompt to analyze the differences
        technique_name = "Tree of Thoughts" if enhanced_type == "ToT" else "Chain-of-Thought"
        analysis_prompt = f"""
        Analyze the following two AI responses to the query: "{user_prompt}"
        
        RESPONSE 1 (Standard prompting):
        {standard_text}
        
        RESPONSE 2 ({technique_name} prompting):
        {enhanced_text}
        
        Please compare these responses considering:
        1. Depth of reasoning
        2. Clarity of explanation
        3. Accuracy of information (if applicable)
        4. Structure and organization
        5. Key differences in approach
        6. Problem-solving methodology
        
        Provide a concise, balanced analysis highlighting the strengths and weaknesses of each approach.
        """
        
        try:
            # Get Bedrock client
            bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
            
            # System prompt for analysis
            system_prompt = [{"text": "You are an expert in analyzing AI responses and prompt engineering. Provide clear, insightful, and balanced comparisons."}]
            
            # Message structure
            message = {
                "role": "user",
                "content": [{"text": analysis_prompt}]
            }
            messages = [message]
            
            # Get analysis from the model
            analysis_params = {
                "temperature": 0.3,
                "topP": 0.9,
                "maxTokens": 1500
            }
            
            analysis_response = text_conversation(
                bedrock_client, 
                "anthropic.claude-3-sonnet-20240229-v1:0", 
                system_prompt, 
                messages, 
                **analysis_params
            )
            
            if analysis_response:
                analysis_text = analysis_response['output']['message']['content'][0].get('text', '')
                st.markdown(analysis_text)
            else:
                st.error("Failed to generate analysis. Please try again.")
                
        except Exception as e:
            st.error(f"Error generating analysis: {str(e)}")
        
        # Display token metrics comparison
        st.markdown("### Token Usage Comparison")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Input Tokens",
                value=standard_tokens['inputTokens'],
                delta=enhanced_tokens['inputTokens'] - standard_tokens['inputTokens'],
                delta_color="inverse"
            )
            
        with col2:
            st.metric(
                label="Output Tokens",
                value=standard_tokens['outputTokens'],
                delta=enhanced_tokens['outputTokens'] - standard_tokens['outputTokens'],
                delta_color="inverse"
            )
            
        with col3:
            st.metric(
                label="Total Tokens",
                value=standard_tokens['totalTokens'],
                delta=enhanced_tokens['totalTokens'] - standard_tokens['totalTokens'],
                delta_color="inverse"
            )
            
        technique_abbrev = "ToT" if enhanced_type == "ToT" else "CoT"
        st.caption(f"Note: Delta shows difference between {technique_abbrev} and Standard ({technique_abbrev} - Standard)")
        
        st.markdown("</div>", unsafe_allow_html=True)

def zero_shot_tab(model_id, params):
    """Content for Zero-Shot tab."""
    with st.expander("Learn more about Zero-Shot Chain-of-Thought", expanded=False):
        st.markdown("### Zero-Shot Chain-of-Thought")
        st.markdown("""
        <div class="key-benefit">
        Zero-Shot CoT simply adds an instruction like "Think step by step" to the prompt, 
        encouraging the model to break down its reasoning process without providing examples.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Key Benefits:**
        - ✅ Minimal additional prompt engineering required
        - ✅ Works surprisingly well with capable models
        - ✅ No need to craft elaborate examples
        - ✅ Can be applied to virtually any reasoning task
        """)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.zero_shot_system_prompt,
        height=80,
        help="This defines how the AI assistant should behave",
        key="zero_shot_system"
    )
    st.session_state.zero_shot_system_prompt = system_prompt
    
    # Callback function for when sample prompt changes
    def on_zero_shot_prompt_change():
        selected_name = st.session_state.zero_shot_select
        selected_prompt = next(p for p in ZERO_SHOT_PROMPTS if p["name"] == selected_name)
        st.session_state.zero_shot_prompt = selected_prompt["prompt"]
    
    # Display sample prompts
    selected_prompt = display_sample_prompts(
        ZERO_SHOT_PROMPTS, 
        st.session_state.zero_shot_prompt,
        "zero_shot",
        on_change_callback=on_zero_shot_prompt_change
    )
    
    # User prompt input
    user_prompt = st.text_area(
        "Prompt", 
        height=120,
        placeholder="Enter your question or task here...",
        key="zero_shot_prompt"
    )
    
    # CoT suffix
    cot_suffix = selected_prompt["cot_suffix"]
    
    # Generate responses and analyze buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button(
            "📝 Generate Responses",
            type="primary",
            key="zero_shot_generate",
            disabled=st.session_state.processing or not user_prompt.strip()
        )
    
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Differences",
            key="zero_shot_analyze",
            disabled=not (st.session_state.zero_shot_standard_response and st.session_state.zero_shot_cot_response) or st.session_state.processing,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process generate button click
    if generate_button and user_prompt.strip() and not st.session_state.processing:
        st.session_state.processing = True
        
        # Standard prompt processing
        with st.status("Generating standard response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for standard prompt
                system_prompts = [{"text": system_prompt}]
                standard_message = {
                    "role": "user",
                    "content": [{"text": user_prompt}]
                }
                standard_messages = [standard_message]
                
                # Send request to the model
                st.session_state.zero_shot_standard_response = text_conversation(
                    bedrock_client, model_id, system_prompts, standard_messages, **params
                )
                
                status.update(label="Standard response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        # CoT prompt processing
        with st.status("Generating Chain-of-Thought response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for CoT prompt
                system_prompts = [{"text": system_prompt}]
                cot_message = {
                    "role": "user",
                    "content": [{"text": user_prompt + cot_suffix}]
                }
                cot_messages = [cot_message]
                
                # Send request to the model
                st.session_state.zero_shot_cot_response = text_conversation(
                    bedrock_client, model_id, system_prompts, cot_messages, **params
                )
                
                status.update(label="Chain-of-Thought response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        st.session_state.processing = False
        st.rerun()
    
    # Process analyze button click
    if analyze_button and st.session_state.zero_shot_standard_response and st.session_state.zero_shot_cot_response:
        st.session_state.analysis_shown = True
        st.rerun()
    
    # Display the responses
    display_responses(st.session_state.zero_shot_standard_response, st.session_state.zero_shot_cot_response)
    
    # Display analysis if available
    display_analysis(
        st.session_state.zero_shot_standard_response, 
        st.session_state.zero_shot_cot_response, 
        user_prompt, 
        st.session_state.analysis_shown
    )

def few_shot_tab(model_id, params):
    """Content for Few-Shot tab."""
    with st.expander("Learn more about Few-Shot Chain-of-Thought", expanded=False):
        st.markdown("### Few-Shot Chain-of-Thought")
        st.markdown("""
        <div class="key-benefit">
        Few-Shot CoT provides explicit examples of step-by-step reasoning before asking the model 
        to solve a new problem. This approach demonstrates the expected reasoning pattern.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Key Benefits:**
        - ✅ Creates a clear pattern for the model to follow
        - ✅ Demonstrates the depth of reasoning expected
        - ✅ Can guide specific reasoning styles or approaches
        - ✅ Often produces more consistent reasoning quality
        """)   
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.few_shot_system_prompt,
        height=80,
        help="This defines how the AI assistant should behave",
        key="few_shot_system"
    )
    st.session_state.few_shot_system_prompt = system_prompt
    
    # Callback function for when sample prompt changes
    def on_few_shot_prompt_change():
        selected_name = st.session_state.few_shot_select
        selected_prompt = next(p for p in FEW_SHOT_PROMPTS if p["name"] == selected_name)
        st.session_state.few_shot_prompt = selected_prompt["prompt"]
        st.session_state.few_shot_cot_prompt = selected_prompt["prompt"] + selected_prompt["cot_suffix"]
    
    # Display sample prompts
    selected_prompt = display_sample_prompts(
        FEW_SHOT_PROMPTS, 
        st.session_state.few_shot_prompt,
        "few_shot",
        on_change_callback=on_few_shot_prompt_change
    )

    # Few-shot prompts (standard and CoT)
    st.markdown("#### Standard Few-Shot")
    few_shot_prompt = st.text_area(
        "Standard Few-Shot Prompt",
        height=300,
        key="few_shot_prompt"
    )
    
    st.markdown("#### Few-Shot with Chain-of-Thought")
    few_shot_cot_prompt = st.text_area(
        "Few-Shot CoT Prompt",
        height=300,
        key="few_shot_cot_prompt"
    )
    
    # Generate responses and analyze buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button(
            "📝 Generate Responses",
            type="primary",
            key="few_shot_generate",
            disabled=st.session_state.processing or not few_shot_prompt.strip()
        )
    
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Differences",
            key="few_shot_analyze",
            disabled=not (st.session_state.few_shot_standard_response and st.session_state.few_shot_cot_response) or st.session_state.processing,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process generate button click
    if generate_button and few_shot_prompt.strip() and not st.session_state.processing:
        st.session_state.processing = True
        
        # Standard prompt processing
        with st.status("Generating standard few-shot response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for standard prompt
                system_prompts = [{"text": system_prompt}]
                standard_message = {
                    "role": "user",
                    "content": [{"text": few_shot_prompt}]
                }
                standard_messages = [standard_message]
                
                # Send request to the model
                st.session_state.few_shot_standard_response = text_conversation(
                    bedrock_client, model_id, system_prompts, standard_messages, **params
                )
                
                status.update(label="Standard few-shot response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        # CoT prompt processing
        with st.status("Generating few-shot Chain-of-Thought response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for CoT prompt
                system_prompts = [{"text": system_prompt}]
                cot_message = {
                    "role": "user",
                    "content": [{"text": few_shot_cot_prompt}]
                }
                cot_messages = [cot_message]
                
                # Send request to the model
                st.session_state.few_shot_cot_response = text_conversation(
                    bedrock_client, model_id, system_prompts, cot_messages, **params
                )
                
                status.update(label="Few-shot Chain-of-Thought response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        st.session_state.processing = False
        st.rerun()
    
    # Process analyze button click
    if analyze_button and st.session_state.few_shot_standard_response and st.session_state.few_shot_cot_response:
        st.session_state.analysis_shown = True
        st.rerun()
    
    # Display the responses
    display_responses(st.session_state.few_shot_standard_response, st.session_state.few_shot_cot_response)
    
    # Display analysis if available
    display_analysis(
        st.session_state.few_shot_standard_response, 
        st.session_state.few_shot_cot_response, 
        "Custom few-shot prompt", 
        st.session_state.analysis_shown
    )

def tot_tab(model_id, params):
    """Content for Tree of Thoughts tab."""
    with st.expander("Learn more about Tree of Thoughts", expanded=False):
        st.markdown("### Tree of Thoughts (ToT)")
        st.markdown("""
        <div class="key-benefit">
        Tree of Thoughts systematically explores multiple reasoning paths and solution strategies. 
        It encourages the model to consider various approaches, evaluate them, and find the optimal solution.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        **Key Benefits:**
        - ✅ Explores multiple solution paths simultaneously
        - ✅ Compares and evaluates different approaches
        - ✅ Finds optimal solutions through systematic search
        - ✅ Excellent for complex problems with multiple valid approaches
        - ✅ Provides comprehensive analysis of problem space
        """)
    
    # System prompt
    system_prompt = st.text_area(
        "System Prompt", 
        value=st.session_state.tot_system_prompt,
        height=80,
        help="This defines how the AI assistant should behave",
        key="tot_system"
    )
    st.session_state.tot_system_prompt = system_prompt
    
    # Callback function for when sample prompt changes
    def on_tot_prompt_change():
        selected_name = st.session_state.tot_select
        selected_prompt = next(p for p in TOT_PROMPTS if p["name"] == selected_name)
        st.session_state.tot_prompt = selected_prompt["prompt"]
        st.session_state.tot_cot_prompt = selected_prompt["prompt"] + selected_prompt["tot_suffix"]
    
    # Display sample prompts
    selected_prompt = display_sample_prompts(
        TOT_PROMPTS, 
        st.session_state.tot_prompt,
        "tot",
        on_change_callback=on_tot_prompt_change
    )

    # ToT prompts (standard and ToT)
    st.markdown("#### Standard Prompt")
    tot_prompt = st.text_area(
        "Standard Prompt",
        height=150,
        key="tot_prompt"
    )
    
    st.markdown("#### Tree of Thoughts Prompt")
    tot_enhanced_prompt = st.text_area(
        "Tree of Thoughts Prompt",
        height=400,
        key="tot_cot_prompt"
    )
    
    # Generate responses and analyze buttons
    col1, col2 = st.columns([3, 1])
    with col1:
        generate_button = st.button(
            "📝 Generate Responses",
            type="primary",
            key="tot_generate",
            disabled=st.session_state.processing or not tot_prompt.strip()
        )
    
    with col2:
        analyze_button = st.button(
            "🔍 Analyze Differences",
            key="tot_analyze",
            disabled=not (st.session_state.tot_standard_response and st.session_state.tot_tot_response) or st.session_state.processing,
            use_container_width=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Process generate button click
    if generate_button and tot_prompt.strip() and not st.session_state.processing:
        st.session_state.processing = True
        
        # Standard prompt processing
        with st.status("Generating standard response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for standard prompt
                system_prompts = [{"text": system_prompt}]
                standard_message = {
                    "role": "user",
                    "content": [{"text": tot_prompt}]
                }
                standard_messages = [standard_message]
                
                # Send request to the model
                st.session_state.tot_standard_response = text_conversation(
                    bedrock_client, model_id, system_prompts, standard_messages, **params
                )
                
                status.update(label="Standard response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        # ToT prompt processing
        with st.status("Generating Tree of Thoughts response...") as status:
            try:
                # Get Bedrock client
                bedrock_client = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')
                
                # Setup the system prompts and messages for ToT prompt
                system_prompts = [{"text": system_prompt}]
                tot_message = {
                    "role": "user",
                    "content": [{"text": tot_enhanced_prompt}]
                }
                tot_messages = [tot_message]
                
                # Send request to the model
                st.session_state.tot_tot_response = text_conversation(
                    bedrock_client, model_id, system_prompts, tot_messages, **params
                )
                
                status.update(label="Tree of Thoughts response generated!", state="complete")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {str(e)}")
        
        st.session_state.processing = False
        st.rerun()
    
    # Process analyze button click
    if analyze_button and st.session_state.tot_standard_response and st.session_state.tot_tot_response:
        st.session_state.analysis_shown = True
        st.rerun()
    
    # Display the responses
    display_responses(st.session_state.tot_standard_response, st.session_state.tot_tot_response, "ToT")
    
    # Display analysis if available
    display_analysis(
        st.session_state.tot_standard_response, 
        st.session_state.tot_tot_response, 
        "Tree of Thoughts prompt", 
        st.session_state.analysis_shown,
        "ToT"
    )

def main():
    load_css()
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown("<h1 class='main-header'>Chain-of-Thought Prompting</h1>", unsafe_allow_html=True)
    
    st.markdown("""<div class="info-box">
    Explore and compare how different Chain-of-Thought prompting techniques improve the reasoning 
    capabilities of large language models. Learn about Zero-shot CoT, Few-shot CoT, and Tree-of-Thought approaches.
    </div>""", unsafe_allow_html=True)
    
    # Create a 70/30 layout
    col1, col2 = st.columns([0.7, 0.3])     
        # Get model and parameters from the right column
    with col2:
        model_id, params = parameter_sidebar()

    with col1:
        # Create tabs for zero-shot, few-shot, and tree of thoughts
        tab1, tab2, tab3 = st.tabs(["🎯 Zero-Shot", "🔄 Few-Shot", "🌳 Tree of Thoughts"])
        
        # Populate each tab
        with tab1:
            zero_shot_tab(model_id, params)
        
        with tab2:
            few_shot_tab(model_id, params)
            
        with tab3:
            tot_tab(model_id, params)

    # Footer
    st.markdown("---")
    st.markdown("<div class='footer'>© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved.</div>", 
                unsafe_allow_html=True)

if __name__ == "__main__":
    if 'localhost' in st.context.headers["host"]:
        main()
    else:
        # First check authentication
        is_authenticated = authenticate.login()
        
        # If authenticated, show the main app content
        if is_authenticated:
            main()