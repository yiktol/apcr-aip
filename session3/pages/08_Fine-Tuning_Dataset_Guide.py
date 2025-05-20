# Modern UI for Foundation Model Fine-Tuning Dataset Preparation
import json
import streamlit as st
import pandas as pd
import uuid
import random
from datetime import datetime
import matplotlib.pyplot as plt


# Page configuration for a modern look and feel
st.set_page_config(
    page_title="Foundation Model Fine-Tuning Datasets",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.8rem !important;
        font-weight: 600 !important;
        color: #2563EB !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .section-header {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #3B82F6 !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.8rem !important;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        border-left: 5px solid #F59E0B;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        border-left: 5px solid #10B981;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .requirement-item {
        background-color: #F3F4F6;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        color: #1E40AF !important;
        font-weight: bold;
    }
    .data-schema-container {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    /* Code block styling */
    pre {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
    }
    code {
        font-family: 'Courier New', Courier, monospace !important;
    }
    .download-button {
        background-color: #2563EB;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        text-decoration: none;
    }
    .download-button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with project information
with st.sidebar:
    st.title("Fine-Tuning Guide")
    st.markdown("### Navigation")
    
    st.markdown("""
    - [Dataset Requirements](#dataset-requirements)
    - [Example Datasets](#example-datasets)
    - [Data Preparation](#data-preparation)
    - [Best Practices](#best-practices)
    """)
    
    st.markdown("---")
    
    st.markdown("### Fine-Tuning Models")
    st.markdown("""
    **Supported Model Families:**
    - Amazon Titan
    - Anthropic Claude
    - Cohere Command
    - Mistral AI
    - AI21 Jurassic
    """)
    
    st.markdown("---")
    
    with st.expander("💡 Pro Tips", expanded=False):
        st.markdown("""
        - Start with a small dataset (~100 examples)
        - Ensure consistent formatting across examples
        - Include diverse examples that cover edge cases
        - Validate your dataset before training
        - Monitor training metrics to detect issues
        """)

# Generate example datasets for demonstration purposes
def generate_text_classification_dataset(n=5):
    categories = ["Technology", "Business", "Entertainment", "Sports", "Politics"]
    
    dataset = []
    for _ in range(n):
        category = random.choice(categories)
        
        if category == "Technology":
            text = random.choice([
                "Apple announces new iPhone with improved camera and longer battery life.",
                "Google's latest algorithm update focuses on AI-powered search results.",
                "Microsoft releases security patch for critical Windows vulnerability.",
                "Tesla unveils new electric vehicle with 500-mile range and autonomous features.",
                "Researchers develop quantum computer that solves complex problems in seconds."
            ])
        elif category == "Business":
            text = random.choice([
                "Stock market reaches all-time high amid economic recovery.",
                "Amazon acquires startup specializing in supply chain management.",
                "Oil prices drop following increased production from major exporters.",
                "Federal Reserve announces interest rate hike to combat inflation.",
                "Startup raises $50 million in Series B funding for expansion."
            ])
        elif category == "Entertainment":
            text = random.choice([
                "New superhero movie breaks box office records on opening weekend.",
                "Popular band announces world tour starting next spring.",
                "Streaming platform adds thousands of classic films to its library.",
                "Celebrity couple announces separation after ten years together.",
                "Award-winning director reveals details about upcoming historical drama."
            ])
        elif category == "Sports":
            text = random.choice([
                "Local team wins championship after dramatic overtime victory.",
                "Star player signs record-breaking contract extension.",
                "Olympic committee announces new sports for 2028 games.",
                "Coach fired after team's disappointing season performance.",
                "Athlete breaks world record that stood for over two decades."
            ])
        else:  # Politics
            text = random.choice([
                "President signs new legislation addressing climate change.",
                "Opposition party criticizes government's economic policies.",
                "Elections result in unexpected shift in parliamentary control.",
                "Diplomatic talks aim to ease tensions between neighboring countries.",
                "Poll shows public opinion divided on proposed healthcare reforms."
            ])
            
        dataset.append({
            "id": str(uuid.uuid4()),
            "text": text,
            "label": category
        })
    
    return dataset

def generate_qa_dataset(n=5):
    questions = [
        "What is Amazon Bedrock?",
        "How do foundation models work?",
        "What is fine-tuning in machine learning?",
        "What's the difference between pre-training and fine-tuning?",
        "How much data is needed for fine-tuning?",
        "What is prompt engineering?",
        "How do I evaluate my fine-tuned model?",
        "What is catastrophic forgetting?",
        "Can fine-tuning remove model biases?",
        "What are the costs associated with fine-tuning?"
    ]
    
    answers = [
        "Amazon Bedrock is a fully managed service that offers high-performing foundation models from leading AI companies through a single API, along with tools to build generative AI applications.",
        "Foundation models are large AI systems trained on vast amounts of data that can be adapted to a wide range of tasks. They use transformer architectures to understand and generate content across different domains.",
        "Fine-tuning is the process of taking a pre-trained model and further training it on a smaller, task-specific dataset to adapt its capabilities for specialized applications.",
        "Pre-training involves training a model on large, general datasets to learn broad patterns and representations, while fine-tuning adapts that pre-trained model to specific tasks with smaller, targeted datasets.",
        "The amount of data needed varies by task complexity, but typically ranges from hundreds to thousands of examples. For simple tasks, even 100-500 high-quality examples can produce good results.",
        "Prompt engineering is the practice of designing effective inputs or instructions that guide AI models to produce desired outputs, often using specific formats or techniques to improve response quality.",
        "Evaluation involves testing the model with validation data, comparing performance metrics against baseline models, and collecting human feedback on output quality, relevance, and accuracy.",
        "Catastrophic forgetting occurs when a model loses previously learned knowledge after being fine-tuned on new data, essentially 'forgetting' its pre-trained capabilities while adapting to new tasks.",
        "Fine-tuning can help reduce some biases by exposing the model to more balanced data, but it cannot eliminate all biases and may sometimes amplify existing ones if the training data is not carefully curated.",
        "Fine-tuning costs include computational resources for training, data preparation and annotation, model storage, and ongoing inference costs when deploying the model."
    ]
    
    dataset = []
    for i in range(min(n, len(questions))):
        dataset.append({
            "id": str(uuid.uuid4()),
            "question": questions[i],
            "answer": answers[i]
        })
    
    return dataset

def generate_summarization_dataset(n=5):
    articles = [
        {
            "article": "Researchers at the University of Technology have discovered a new method for carbon capture that could significantly reduce greenhouse gas emissions. The breakthrough involves a novel material that can absorb carbon dioxide from industrial exhaust with 40% greater efficiency than current methods. According to lead scientist Dr. Sarah Chen, the material can be produced at low cost and integrated into existing industrial infrastructure. 'This could be a game-changer for industries struggling to meet emissions targets,' Chen explained. The team has filed for a patent and is working with several manufacturing companies to implement pilot programs within the next year. The research was funded by the National Science Foundation and published in the latest issue of the Journal of Environmental Innovation.",
            "summary": "Researchers developed a new material for carbon capture that is 40% more efficient than current methods, is cost-effective, and can be integrated with existing infrastructure. The team is pursuing patents and industry partnerships for implementation."
        },
        {
            "article": "The city council voted 7-2 last night to approve the downtown revitalization plan after a contentious four-hour meeting. The $28 million project will transform the abandoned industrial district into a mixed-use development with affordable housing units, retail spaces, and a public park. Supporters argued that the plan would create jobs and address the city's housing shortage, while opponents expressed concerns about gentrification and increased traffic. Mayor Rodriguez called the vote 'a historic step forward' for the city. Construction is scheduled to begin in March and will be completed in phases over three years. The project will be funded through a combination of federal grants, municipal bonds, and private investment from regional developers.",
            "summary": "City council approved a $28M downtown revitalization plan with a 7-2 vote that will convert an abandoned industrial area into mixed-use development. The project includes housing, retail, and a park, funded by grants, bonds, and private investment, with construction starting in March."
        },
        {
            "article": "A comprehensive study published yesterday in the International Journal of Nutrition reveals that incorporating fermented foods into daily diets can significantly improve gut health and immune function. The research, which followed 4,500 participants across six countries over two years, found that consuming at least one serving of fermented foods daily was associated with a 28% reduction in inflammatory markers and improved microbiome diversity. The effects were consistent across different age groups and geographical locations. 'What's remarkable is how quickly these benefits appeared in the data - within just two weeks of dietary changes,' said Dr. Miguel Santos, the study's principal investigator. The researchers recommend foods such as yogurt, kimchi, sauerkraut, and kombucha as accessible options for most consumers. The findings could influence future dietary guidelines and preventative health recommendations worldwide.",
            "summary": "A two-year international study with 4,500 participants found that daily consumption of fermented foods reduces inflammatory markers by 28% and improves microbiome diversity. Benefits appeared within two weeks across all age groups and locations, suggesting potential changes to dietary guidelines."
        },
        {
            "article": "The annual Global Technology Summit concluded yesterday with industry leaders predicting significant advances in artificial intelligence regulation within the next year. Representatives from major tech companies, government agencies, and academic institutions gathered to discuss the ethical implications of AI deployment across sectors. Keynote speaker and former regulatory chief Emma Wilson emphasized the need for international cooperation on AI governance: 'We're at an inflection point where the technology is outpacing our regulatory frameworks.' Several countries announced plans to establish specialized AI ethics committees, while a consortium of tech giants pledged $300 million toward research on potential risks and safeguards. The summit also highlighted concerns about AI-related job displacement, with economists presenting conflicting models on long-term employment impacts. Next year's summit is scheduled to take place in Singapore with a focus on AI applications in healthcare and climate science.",
            "summary": "The Global Technology Summit concluded with predictions of imminent AI regulation advances. Leaders discussed ethical implications of AI across sectors, with countries planning specialized ethics committees and tech companies pledging $300M for risk research. Job displacement concerns were raised, and next year's summit will focus on AI in healthcare and climate science."
        },
        {
            "article": "A groundbreaking archaeological discovery in the highlands of Peru has uncovered evidence of an unknown pre-Incan civilization dating back approximately 5,000 years. The excavation, led by Dr. Lucia Ramirez of the National University, revealed an extensive complex of stone structures, advanced irrigation systems, and ceremonial artifacts that don't match any previously documented cultures in the region. Carbon dating of organic materials found at the site confirms the settlement predates other known Andean civilizations by several centuries. 'This challenges our entire chronology of early South American societies,' Ramirez stated at the press conference announcing the findings. The team also discovered unusual astronomical alignments in the main structures, suggesting sophisticated knowledge of celestial movements. The site, which spans approximately 30 hectares, has been secured for continued research, with plans for international collaboration to fully analyze the artifacts and architectural features.",
            "summary": "Archaeologists discovered evidence of an unknown 5,000-year-old pre-Incan civilization in Peru, featuring stone structures, advanced irrigation, and ceremonial artifacts that predate known Andean cultures. The 30-hectare site includes astronomical alignments suggesting advanced celestial knowledge, challenging existing chronologies of South American societies."
        }
    ]
    
    dataset = []
    for i in range(min(n, len(articles))):
        dataset.append({
            "id": str(uuid.uuid4()),
            "article": articles[i]["article"],
            "summary": articles[i]["summary"]
        })
    
    return dataset

def generate_instruction_dataset(n=5):
    examples = [
        {
            "instruction": "Create a weekly meal plan for a vegetarian family of four.",
            "input": "Include breakfast, lunch, and dinner. Focus on protein-rich foods and kid-friendly options. One family member has a nut allergy.",
            "output": "# Weekly Vegetarian Meal Plan for Family of Four (Nut-Free)\n\n## Monday\n**Breakfast:** Greek yogurt parfaits with berries and seed granola\n**Lunch:** Lentil soup with whole grain bread and cheese\n**Dinner:** Black bean and vegetable enchiladas with avocado salad\n\n## Tuesday\n**Breakfast:** Tofu scramble with vegetables and whole grain toast\n**Lunch:** Chickpea salad sandwiches with cucumber slices\n**Dinner:** Vegetable stir-fry with tofu and brown rice\n\n## Wednesday\n**Breakfast:** Oatmeal with sunflower butter, apple slices, and cinnamon\n**Lunch:** Pasta salad with white beans, vegetables, and Italian dressing\n**Dinner:** Homemade vegetable pizza with side salad\n\n## Thursday\n**Breakfast:** Smoothie bowls with spinach, banana, and sunflower seeds\n**Lunch:** Baked potato bar with various toppings (beans, cheese, vegetables)\n**Dinner:** Lentil shepherd's pie with steamed broccoli\n\n## Friday\n**Breakfast:** Breakfast burritos with eggs, cheese, and vegetables\n**Lunch:** Tomato soup with grilled cheese sandwiches\n**Dinner:** Chickpea curry with basmati rice and cucumber raita\n\n## Saturday\n**Breakfast:** Whole grain pancakes with fruit compote and yogurt\n**Lunch:** Mediterranean mezze platter (hummus, falafel, vegetables, olives, pita)\n**Dinner:** Eggplant parmesan with garlic bread and side salad\n\n## Sunday\n**Breakfast:** Vegetable frittata with whole grain toast\n**Lunch:** Quinoa bowls with roasted vegetables and tahini dressing\n**Dinner:** Bean and vegetable chili with cornbread\n\n**Snack options throughout the week:**\n- Fresh fruit\n- Vegetable sticks with hummus\n- Cheese and crackers\n- Sunflower seed butter and apple slices\n- Popcorn\n- Yogurt with honey"
        },
        {
            "instruction": "Explain quantum computing to a high school student.",
            "input": "",
            "output": "# Quantum Computing Explained\n\nImagine your regular computer, like your laptop or smartphone. It processes information using bits, which are like tiny switches that can be either ON (1) or OFF (0). Everything your computer does—from showing videos to sending messages—comes down to manipulating these bits.\n\nQuantum computers work with something wildly different called quantum bits or \"qubits.\" Here's where it gets interesting!\n\n## What Makes Qubits Special?\n\nWhile regular bits can only be 0 OR 1, qubits can exist in a state that's somehow both 0 AND 1 at the same time. This is called \"superposition\"—think of it like a coin spinning in the air that's neither heads nor tails until it lands.\n\nEven cooler, qubits can be \"entangled\" with each other. When two qubits are entangled, whatever happens to one instantly affects the other, even if they're separated by a great distance. Einstein called this \"spooky action at a distance.\"\n\n## What Can Quantum Computers Do?\n\nBecause of these unique properties, quantum computers can process certain types of problems much faster than regular computers:\n\n- They could crack codes that would take normal computers thousands of years to solve\n- They might help design new medicines by simulating molecules with perfect accuracy\n- They could optimize complex systems like city traffic or global shipping routes\n\n## The Challenges\n\nQuantum computers aren't replacing your laptop anytime soon because:\n\n1. They're extremely sensitive—even tiny vibrations or temperature changes can cause errors\n2. They need to be kept colder than outer space to work properly\n3. We're still learning how to program them effectively\n\n## The Big Picture\n\nThink of regular computers as being really good at following precise step-by-step instructions, while quantum computers are better at finding patterns and solutions in enormous amounts of data all at once.\n\nWhile still in their early stages, quantum computers represent a fundamentally new way of processing information that could eventually help us solve problems we can't even approach with today's technology!"
        },
        {
            "instruction": "Write a cover letter for a software engineering job application.",
            "input": "Job requirements: 5+ years of Python development, experience with cloud platforms, strong problem-solving skills. My background: 6 years as a Python developer at a fintech company, AWS certified, implemented machine learning solutions that reduced fraud by 40%.",
            "output": "Dear Hiring Manager,\n\nI am writing to express my interest in the Software Engineering position at your company. With six years of experience as a Python developer in the fintech industry and AWS certification, I believe my technical expertise and problem-solving abilities align perfectly with your team's needs.\n\nIn my current role at FinTech Solutions Inc., I've led the development of several critical Python applications that form the backbone of our fraud detection system. Most notably, I architected and implemented machine learning models that successfully reduced fraud incidents by 40%, resulting in millions of dollars in saved revenue. This project required not only deep Python knowledge but also the ability to design scalable solutions on AWS infrastructure.\n\nMy day-to-day responsibilities involve:\n- Developing and maintaining complex Python codebases in a collaborative environment\n- Designing and implementing cloud-native applications on AWS\n- Creating robust APIs that handle millions of transactions daily\n- Mentoring junior developers and establishing best practices\n\nI'm particularly drawn to your company because of your reputation for innovation and your commitment to solving challenging technical problems. The opportunity to apply my Python expertise and cloud knowledge to new domains excites me, and I'm confident that my problem-solving approach would bring valuable perspectives to your team.\n\nBeyond technical qualifications, I pride myself on being a clear communicator who thrives in collaborative environments. I believe that the best software emerges from teams that can effectively share ideas and work toward common goals.\n\nI would welcome the opportunity to discuss how my experience could benefit your team. Thank you for considering my application, and I look forward to speaking with you soon.\n\nSincerely,\n[Your Name]"
        },
        {
            "instruction": "Explain the process of photosynthesis and its importance in ecosystems.",
            "input": "",
            "output": "# Photosynthesis: Nature's Solar Power System\n\n## The Basic Process\n\nPhotosynthesis is the remarkable process that allows plants, algae, and some bacteria to convert sunlight, water, and carbon dioxide into glucose (sugar) and oxygen. Think of it as nature's solar power system—capturing energy from sunlight and storing it in chemical bonds.\n\nThe simplified equation looks like this:\n\n6 CO₂ + 6 H₂O + Light Energy → C₆H₁₂O₆ + 6 O₂\n(Carbon Dioxide + Water + Light → Glucose + Oxygen)\n\n## How It Actually Works\n\nPhotosynthesis occurs primarily in the leaf cells of plants, specifically within tiny organelles called chloroplasts that contain the green pigment chlorophyll. The process has two main stages:\n\n1. **Light-Dependent Reactions:**\n   - Occur in the thylakoid membranes of chloroplasts\n   - Chlorophyll captures light energy\n   - Water molecules are split, releasing oxygen as a byproduct\n   - Light energy is converted to chemical energy in the form of ATP and NADPH\n\n2. **Calvin Cycle (Light-Independent Reactions):**\n   - Takes place in the stroma of chloroplasts\n   - Uses the ATP and NADPH from the first stage\n   - Carbon dioxide is incorporated into organic molecules\n   - Glucose and other carbohydrates are produced\n\n## Ecological Importance\n\nPhotosynthesis is absolutely fundamental to life on Earth for several critical reasons:\n\n### 1. Base of Food Webs\n\nPlants create their own food through photosynthesis, making them primary producers. They form the foundation of nearly all food chains and webs, directly or indirectly feeding almost every organism on the planet.\n\n### 2. Oxygen Production\n\nPhotosynthesis is responsible for producing the vast majority of atmospheric oxygen that animals (including humans) breathe. Historically, it transformed Earth's early atmosphere and made complex life possible.\n\n### 3. Carbon Cycle Regulation\n\nPlants absorb carbon dioxide during photosynthesis, making them crucial carbon sinks that help regulate the global carbon cycle and mitigate climate change. Forests, grasslands, and ocean phytoplankton collectively remove billions of tons of carbon dioxide from the atmosphere annually.\n\n### 4. Habitat and Resources\n\nPhotosynthesizing organisms create complex physical structures (like forests) that serve as habitats for countless species. They also produce not just food but materials like wood, fibers, and compounds used for medicine.\n\n## Global Impact\n\nThe significance of photosynthesis extends to global systems:\n- It drives weather patterns through its influence on atmospheric composition\n- It affects water cycles through transpiration\n- It stabilizes soil and prevents erosion\n- It provides renewable resources for human societies\n\nIn essence, photosynthesis connects the sun's energy to Earth's living systems, creating the conditions that allow for the incredible biodiversity we see today. Without this process, life as we know it would be impossible."
        },
        {
            "instruction": "Create a detailed marketing plan for launching a new mobile fitness app.",
            "input": "Target audience: Adults 25-40 who want convenient home workouts. App features include personalized training plans, progress tracking, and community challenges. Budget: $50,000.",
            "output": "# Marketing Plan: FitLife Mobile App Launch\n\n## Executive Summary\n\nThis marketing plan outlines the strategy for launching FitLife, a new mobile fitness application targeting adults aged 25-40 seeking convenient home workout solutions. With a $50,000 budget, the plan focuses on digital marketing channels, strategic partnerships, and community building to achieve maximum market penetration and user acquisition.\n\n## Target Audience Analysis\n\n### Primary Demographic\n- **Age Range:** 25-40 years\n- **Lifestyle:** Busy professionals, parents with limited time\n- **Pain Points:** Gym accessibility, scheduling constraints, lack of personalized guidance\n- **Motivations:** Convenience, flexibility, measurable results, community support\n\n### Audience Segments\n1. **Fitness Beginners:** Need guidance and encouragement\n2. **Former Athletes:** Looking to maintain fitness despite busy schedules\n3. **Work-from-Home Professionals:** Seeking integration of fitness into daily routine\n4. **Parents:** Need time-efficient workouts that can be done at home\n\n## Competitive Analysis\n\n| Competitor | Strengths | Weaknesses | Our Advantage |\n|------------|-----------|------------|---------------|\n| FitBit App | Hardware integration, established brand | Limited personalization | More customized training plans |\n| Nike Training Club | Brand recognition, production quality | Limited community features | Stronger social elements |\n| Peloton Digital | Premium content, loyal following | High price point | More affordable, no equipment required |\n| MyFitnessPal | Large user base, nutrition focus | Basic workout features | Superior workout personalization |\n\n## Marketing Strategy\n\n### Pre-Launch Phase (4-6 weeks before launch)\n\n**Activities:**\n- Create teaser website with email signup (Budget: $1,500)\n- Recruit 50-100 beta testers from target demographic (Budget: $2,000)\n- Develop influencer outreach program (Budget: $1,000)\n- Create content calendar for first 3 months (Budget: $500)\n\n**Goals:**\n- Collect 5,000+ email subscribers\n- Generate initial social media presence with 1,000+ followers\n- Refine app based on beta feedback\n\n### Launch Phase (1-2 weeks)\n\n**Activities:**\n- Influencer partnerships with 10-15 micro-influencers (Budget: $15,000)\n- Paid social media campaign across Instagram, Facebook, TikTok (Budget: $10,000)\n- Search engine marketing focused on fitness keywords (Budget: $5,000)\n- PR outreach to fitness and technology publications (Budget: $3,000)\n- Launch email sequence to pre-launch subscribers (Budget: $500)\n\n**Goals:**\n- Achieve 20,000 app downloads in first two weeks\n- Generate 50+ media mentions\n- Maintain 4.5+ star rating in app stores\n\n### Post-Launch Phase (Months 1-3)\n\n**Activities:**\n- Ongoing social media content and community management (Budget: $3,000)\n- Referral program implementation and promotion (Budget: $2,500)\n- Continued SEM and social media advertising (Budget: $5,000)\n- First community challenge event (Budget: $1,000)\n\n**Goals:**\n- Reach 50,000 total downloads by end of month 3\n- Achieve 40% user retention rate after 30 days\n- Generate 500+ user-generated content pieces\n\n## Content Strategy\n\n### Content Themes\n1. Success stories and transformations\n2. Quick workout tips and demonstrations\n3. Behind-the-scenes app development\n4. Fitness myths debunked\n5. Lifestyle integration content\n\n### Content Formats\n- Short-form videos (15-60 seconds)\n- Before/after imagery\n- Workout infographics\n- Expert Q&A sessions\n- User testimonials\n\n## Measurement & KPIs\n\n### Primary KPIs\n- App downloads\n- User retention rates (7-day, 30-day)\n- Daily/weekly active users\n- In-app engagement metrics\n- Cost per acquisition\n\n### Secondary KPIs\n- Social media engagement rates\n- Email open and click rates\n- Website traffic\n- Media mentions\n- App store rating\n\n## Budget Allocation\n\n| Category | Amount | Percentage |\n|----------|--------|------------|\n| Influencer Marketing | $15,000 | 30% |\n| Paid Advertising | $20,000 | 40% |\n| PR & Media | $3,000 | 6% |\n| Content Creation | $5,000 | 10% |\n| Beta Testing & Research | $2,000 | 4% |\n| Website & Technical | $2,000 | 4% |\n| Community Building | $1,500 | 3% |\n| Contingency | $1,500 | 3% |\n| **TOTAL** | **$50,000** | **100%** |\n\n## Timeline\n\n**Weeks 1-4:** Pre-launch activities, beta testing, content creation\n**Weeks 5-6:** Launch preparations, influencer briefings, ad creative development\n**Week 7:** Official Launch, heavy promotion\n**Weeks 8-12:** Post-launch optimization, community building\n**Week 16:** First major in-app community challenge\n\n## Risk Assessment & Mitigation\n\n| Risk | Impact | Mitigation Strategy |\n|------|--------|---------------------|\n| Lower than expected downloads | High | Increase ad spend, adjust targeting |\n| Poor retention rates | High | Analyze drop-off points, implement improvements |\n| Negative reviews | Medium | Rapid response team, prioritize fixes |\n| Competitor response | Medium | Monitor competitor activity, emphasize unique features |\n| Budget overrun | Medium | 3% contingency, biweekly budget reviews |\n\n## Conclusion\n\nThis marketing plan leverages digital channels and community-building to establish FitLife as a convenient, effective home workout solution. The focus on influencer partnerships and user testimonials will build credibility, while ongoing engagement activities will support retention. With careful execution and monitoring of KPIs, we can achieve our launch goals within the $50,000 budget constraint."
        }
    ]
    
    dataset = []
    for i in range(min(n, len(examples))):
        dataset.append({
            "id": str(uuid.uuid4()),
            "instruction": examples[i]["instruction"],
            "input": examples[i]["input"],
            "output": examples[i]["output"]
        })
    
    return dataset

def generate_chat_dataset(n=5):
    conversations = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful customer service assistant for a software company."},
                {"role": "user", "content": "I'm having trouble logging into my account. I keep getting an 'invalid credentials' error."},
                {"role": "assistant", "content": "I'm sorry to hear you're having trouble logging in. Let's troubleshoot this together. There are a few common reasons for 'invalid credentials' errors:\n\n1. Incorrect username or password\n2. Caps Lock might be turned on\n3. Password might have expired\n4. Account might be locked after too many failed attempts\n\nCould you try resetting your password first? You can do this by clicking the 'Forgot Password' link on the login page. If that doesn't work, please let me know and I can check if there are any issues with your account."},
                {"role": "user", "content": "I tried resetting my password but I'm not receiving the reset email."},
                {"role": "assistant", "content": "Thank you for trying that step. If you're not receiving the password reset email, here are some things we can check:\n\n1. Please check your spam or junk folder in case the email was filtered there.\n2. Verify that you're using the correct email address associated with your account.\n3. Some email providers might delay delivery, so please allow up to 15 minutes for the email to arrive.\n\nIf you've checked these and still don't see the email, I can manually reset your password or send the reset link to an alternative email address. To do this, I'll need to verify your identity. Could you please provide your account username and the email address associated with your account?"},
                {"role": "user", "content": "My username is johnsmith92 and my email is john.smith@example.com"},
                {"role": "assistant", "content": "Thank you for providing that information, Mr. Smith. I've checked your account and I notice a couple of things:\n\n1. Your account is active, so there are no restrictions that would prevent you from logging in.\n2. I see that your email domain (example.com) is sometimes flagged by our email delivery system.\n\nI've taken two actions to help you:\n\n1. I've manually triggered a new password reset email to john.smith@example.com. Please check again in a few minutes, including your spam folder.\n2. I've added a temporary alternate recovery email option. If you'd like to use it, I can send the reset link to a different email address.\n\nPlease let me know if you receive the reset email this time or if you'd prefer to use an alternate email address."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an AI language tutor helping students learn Spanish. Be encouraging and provide explanations for corrections."},
                {"role": "user", "content": "Hola! Me llamo David. Quiero practicar mi español. Yo soy estudiante."},
                {"role": "assistant", "content": "¡Hola David! Encantado de conocerte. Tu español suena muy bien. Has formado estas frases correctamente:\n\n- \"Me llamo David\" ✓\n- \"Quiero practicar mi español\" ✓\n- \"Yo soy estudiante\" ✓\n\nUna pequeña sugerencia: en \"Yo soy estudiante\" el pronombre \"Yo\" es técnicamente correcto, pero en español a menudo omitimos los pronombres personales cuando el contexto es claro. Así que podrías simplemente decir \"Soy estudiante\".\n\n¿De qué te gustaría hablar para practicar más tu español?"},
                {"role": "user", "content": "Gracias! Yo quiero hablar sobre mi familia. Tengo dos hermanos y uno hermana. Mi padre es profesor y mi madre es doctora."},
                {"role": "assistant", "content": "¡Muy bien, David! Me encanta que quieras hablar sobre tu familia. Tu español fluye bastante bien.\n\nPermíteme hacer un par de pequeñas correcciones:\n\n- \"Tengo dos hermanos y **una** hermana\" - La palabra \"hermana\" es femenina, así que necesitamos usar \"una\" en lugar de \"uno\".\n- Como mencioné antes, podrías omitir \"Yo\" en \"Yo quiero hablar\" y simplemente decir \"Quiero hablar\".\n\nEl resto está perfecto:\n- \"Mi padre es profesor\" ✓\n- \"Mi madre es doctora\" ✓\n\n¡Qué interesante familia tienes! ¿Tus hermanos también son estudiantes? ¿Y qué estudias tú?"},
                {"role": "user", "content": "Estudio informática en la universidad. Mis hermanos son más jóvenes que yo. Ellos están en la escuela secundaria. ¿Puede explicarme cuando usar 'ser' y cuando usar 'estar'?"},
                {"role": "assistant", "content": "¡Excelente, David! Tu español está mejorando mucho. Todas las frases en tu último mensaje están perfectas gramaticalmente. Estudiar informática suena interesante.\n\nLa distinción entre \"ser\" y \"estar\" es uno de los aspectos más desafiantes del español. Aquí te explico la diferencia principal:\n\n**Ser** se usa para:\n1. Características permanentes o esenciales: \"Soy alto\" (I am tall)\n2. Profesión: \"Es profesor\" (He is a teacher)\n3. Nacionalidad/origen: \"Somos españoles\" (We are Spanish)\n4. Hora/fecha: \"Es lunes\" (It's Monday)\n5. Material: \"La mesa es de madera\" (The table is made of wood)\n\n**Estar** se usa para:\n1. Estados temporales: \"Estoy cansado\" (I am tired)\n2. Ubicación: \"El libro está en la mesa\" (The book is on the table)\n3. Estados resultantes de un cambio: \"La sopa está caliente\" (The soup is hot)\n4. Acciones en progreso: \"Estoy estudiando\" (I am studying)\n5. Estados de ánimo: \"Estás feliz\" (You are happy)\n\nUn truco para recordar: **SER** para características **E**senciales y p**ER**manentes, **ESTAR** para **EST**ados y ubic**A**ción **T**emporales.\n\n¿Te gustaría practicar con algunos ejemplos?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are an AI assistant helping people plan trips. Provide detailed, helpful travel advice based on the user's preferences."},
                {"role": "user", "content": "I'm planning a 5-day trip to Japan in October. I'm interested in a mix of traditional culture and modern attractions. Any recommendations?"},
                {"role": "assistant", "content": "A 5-day trip to Japan in October is perfect timing! You'll enjoy pleasant autumn weather and possibly see early fall colors in some regions. Here's a suggested itinerary that balances traditional culture with modern experiences:\n\n**Day 1-2: Tokyo**\n- Morning: Visit Senso-ji Temple in Asakusa for traditional culture\n- Afternoon: Explore the modern districts of Shibuya and Shinjuku\n- Evening: Experience Robot Restaurant or a traditional izakaya\n- Next day: Visit the teamLab Borderless digital art museum and the traditional Meiji Shrine\n\n**Day 3: Day trip to Kamakura**\n- See the Great Buddha (Daibutsu)\n- Visit ancient temples like Hase-dera\n- Stroll through the bamboo grove at Hokoku-ji\n\n**Day 4-5: Kyoto**\n- Take an early shinkansen (bullet train) to Kyoto\n- Visit top temples like Kinkaku-ji (Golden Pavilion) and Fushimi Inari Shrine\n- Experience a traditional tea ceremony\n- Explore Gion district and potentially spot geisha\n- Visit Arashiyama Bamboo Grove in the morning before crowds arrive\n\n**October-specific tips:**\n- Check for autumn festivals (matsuri) happening during your visit\n- In mid-to-late October, visit Rikugien Garden in Tokyo or Tofuku-ji in Kyoto for early autumn colors\n- Bring layers as temperatures can range from 15-23°C (59-73°F)\n\n**Modern/traditional combinations I recommend:**\n- Stay in a capsule hotel in Tokyo and a traditional ryokan in Kyoto\n- Try both conveyor belt sushi and a traditional kaiseki meal\n- Visit both Nintendo Tokyo and traditional craft shops\n\nWould you like more specific recommendations for any of these areas or information about transportation between cities?"},
                {"role": "user", "content": "That sounds great! I'm particularly interested in food experiences in Japan. What are some must-try foods and good places to find them?"},
                {"role": "assistant", "content": "Japanese food experiences are incredible! Here's a guide to must-try foods and where to find them during your trip:\n\n## Tokyo Food Experiences:\n\n**Tsukiji Outer Market**\n- While the inner market moved to Toyosu, the outer market remains and is perfect for:\n  - Fresh sushi breakfasts (try Sushi Dai or Daiwa Sushi)\n  - Street food like tamagoyaki (sweet omelet)\n  - Seafood skewers\n\n**Ramen**\n- Tokyo Ramen Street (Tokyo Station): 8 famous ramen shops in one location\n- Ichiran (Shibuya): Famous tonkotsu ramen in individual booths\n- Afuri (multiple locations): Yuzu-flavored lighter ramen\n\n**Izakayas (Japanese pubs)**\n- Omoide Yokocho (Memory Lane/Piss Alley) in Shinjuku: Tiny traditional izakayas\n- Try yakitori (grilled chicken skewers), karaage (fried chicken), and sashimi with sake\n\n**Modern Experiences**\n- Kawaii Monster Cafe (Harajuku): Bizarre, colorful food experience\n- Any depachika (department store food hall): Try incredible takeaway foods at Isetan or Takashimaya\n\n## Kyoto Food Experiences:\n\n**Traditional Kyoto Cuisine**\n- Nishiki Market: Sample Kyoto specialties like tsukemono (pickles) and yuba (tofu skin)\n- Try shojin ryori (Buddhist vegetarian cuisine) at temples like Tenryu-ji\n- Experience kaiseki (multi-course) dining at Gion Karyo or Roan Kikunoi\n\n**Unique to Kyoto**\n- Yudofu (hot tofu): Best at Nanzen-ji temple area restaurants\n- Kyo-wagashi (Kyoto sweets): Try at Kagizen Yoshifusa with matcha tea\n\n## Must-Try Japanese Foods:\n\n1. **Takoyaki**: Octopus balls found at street stands\n2. **Okonomiyaki**: Savory pancake - try in specialty restaurants\n3. **Japanese curry**: Chain restaurants like CoCo Ichibanya are good\n4. **Tempura**: Best at specialized tempura restaurants\n5. **Onigiri**: Rice balls from convenience stores (7-Eleven has surprisingly good ones!)\n6. **Wagyu beef**: Splurge on A5 wagyu at least once\n7. **Japanese breakfast**: Experience at your ryokan or hotel with grilled fish, rice, miso soup\n8. **Matcha desserts**: Try parfaits, ice cream, and other sweets in Kyoto especially\n\n## Food Experiences Worth Booking:\n\n- Tea ceremony in Kyoto (try Camellia in Southern Higashiyama)\n- Sake tasting tour in Tokyo\n- Cooking class (Tokyo Kitchen or Kyoto Cooking Class)\n- Department store food hall tour\n\nWould you like recommendations for specific dietary preferences or budget-friendly options? Or should I suggest some specific restaurants to book in advance?"},
                {"role": "user", "content": "I'm a vegetarian. Will I have trouble finding food options in Japan?"},
                {"role": "assistant", "content": "Being vegetarian in Japan can be challenging but definitely manageable with some preparation! The Japanese concept of vegetarianism differs from Western definitions—many dishes that appear vegetarian might contain fish stock (dashi) or other seafood ingredients.\n\nHere's your vegetarian survival guide for Japan:\n\n## Useful Japanese Phrases\n\nThese will be incredibly helpful:\n- \"Watashi wa bejitarian desu\" (I am vegetarian)\n- \"Niku nashi de onegaishimasu\" (Without meat please)\n- \"Sakana mo taberaremasen\" (I cannot eat fish either)\n- \"Dashi wa daijoubu desu ka?\" (Is there fish stock?)\n\nConsider saving these phrases in Japanese on your phone to show at restaurants.\n\n## Vegetarian-Friendly Japanese Foods\n\n**Tokyo:**\n- Shojin Ryori: Traditional Buddhist vegetarian cuisine at Itosho or Bon\n- Zen: Upscale vegetarian restaurant in Roppongi\n- T's TanTan in Tokyo Station: Vegan ramen restaurant\n- Ain Soph Journey (Shinjuku) and Ain Soph Ripple (Kabukicho): Popular vegan chain\n- Saishoku Vegetarian: Buffet-style vegetarian restaurant\n\n**Kyoto:** (Much more vegetarian-friendly!)\n- Shigetsu at Tenryuji Temple: Authentic shojin ryori experience\n- Mumokuteki Cafe: Modern vegetarian/vegan cafe\n-Choice: Trendy vegan cafe with Western options\n- Soma: Casual vegetarian restaurant with Japanese dishes\n- Many temples offer vegetarian meals (with reservation)\n\n## Convenience Foods\n\n- Convenience stores (konbini) offer:\n  - Inari-zushi (sweet tofu pockets with rice)\n  - Onigiri with umeboshi (pickled plum) or kombu (seaweed)\n  - Various salads (check ingredients)\n\n- Department store food halls have vegetable-focused bento boxes\n\n## Vegetarian-Friendly Japanese Dishes\n\n- Zaru soba/udon (cold noodles) - but request without the dipping sauce (usually fish-based)\n- Tempura vegetables (specify no dashi in the sauce)\n- Vegetable maki rolls (but confirm no bonito flakes)\n- Tamagoyaki (sweet omelet) - when prepared without dashi\n- Vegetable okonomiyaki (need to specify no bonito flakes, dashi, or tempura bits)\n\n## Apps to Download\n\n1. HappyCow - Find vegetarian restaurants nearby\n2. Google Translate with Japanese offline package\n3. Gurunavi or TableAll for restaurant reservations (use filters)\n\n## Tips for Success\n\n- Research and plan meals in advance—spontaneous dining is trickier\n- Stay in accommodations with kitchens so you can cook sometimes\n- International areas of Tokyo (Roppongi, Hiroo) have more veggie options\n- Look for Indian restaurants which often have vegetarian menus\n- Consider bringing some portable snacks for emergencies\n\nWould you like me to recommend a day-by-day vegetarian food itinerary for your specific trip? Or would you prefer information about specific dishes to avoid?"},
                {"role": "user", "content": "That's very helpful information. One last question - what's the easiest way to get from Tokyo to Kyoto?"},
                {"role": "assistant", "content": "# Tokyo to Kyoto: Transportation Guide\n\nThe easiest and most popular way to travel between Tokyo and Kyoto is by Shinkansen (bullet train). Here's a complete breakdown of your options:\n\n## 1. Shinkansen (Bullet Train) - RECOMMENDED\n\n**Overview:**\n- Fastest and most convenient option\n- Travel time: ~2 hours 15 minutes (Nozomi train) or ~2 hours 40 minutes (Hikari train)\n- Extremely punctual and comfortable\n- Trains depart from Tokyo Station and Shinagawa Station to Kyoto Station\n\n**Ticket Options:**\n\na) **Regular Tickets**\n- One-way fare: ~¥14,000 (~$95) per person\n- Purchase at ticket machines or JR ticket offices (Midori-no-madoguchi)\n- Reserved seats recommended, especially during peak times\n\nb) **Japan Rail Pass** (only if you already planned to get one)\n- 7-day pass: ~¥33,000 (~$225)\n- Must be purchased BEFORE arriving in Japan\n- Cannot use the fastest Nozomi trains (only Hikari and Kodama)\n- Worth it if you're making multiple long-distance trips\n\n**Pro Tips:**\n- Trains run frequently (1-3 departures per hour)\n- You can bring luggage on board (limit two large pieces)\n- Consider forwarding large luggage via delivery service for ~¥2,000 per item\n- Book seats on the right side (E seats) from Tokyo to Kyoto for Mt. Fuji views on clear days\n\n## 2. Highway Bus\n\n**Overview:**\n- Budget option but much slower\n- Travel time: 6-8 hours depending on traffic\n- Overnight options available to save on accommodation\n- Departs from Tokyo Station, Shinjuku, etc.\n\n**Cost:**\n- Day bus: ¥4,000-¥5,000 (~$27-$34)\n- Night bus: ¥5,000-¥8,000 (~$34-$54)\n\n**Note:** While cheaper, I don't recommend this option with only 5 days in Japan unless you're on a very tight budget.\n\n## 3. Flights\n\n**Overview:**\n- Flights connect Tokyo Haneda to Osaka Itami\n- Then need additional transport to Kyoto\n- Total travel time including transfers: ~3-4 hours\n- Not recommended unless you have airline status/miles to use\n\n## My Recommendation\n\nFor your 5-day trip, I strongly recommend the Shinkansen for these reasons:\n1. Maximizes your limited time in Japan\n2. Comfortable and stress-free experience\n3. Reliable schedule helps with trip planning\n4. Part of the Japanese travel experience itself\n\n**Booking advice:** If not using a JR Pass, you can book tickets just 1-2 days in advance during regular season. In October (autumn leaves season), I'd suggest booking as soon as you know your travel date.\n\nWould you like me to explain how to book Shinkansen tickets online or through the JR offices? Or do you have any other questions about the train journey?"}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI coding assistant skilled in Python programming."},
                {"role": "user", "content": "I need help creating a Python function that checks if a string is a palindrome."},
                {"role": "assistant", "content": "I'd be happy to help you create a Python function to check if a string is a palindrome!\n\nA palindrome is a word, phrase, or sequence that reads the same backward as forward (ignoring spaces, punctuation, and capitalization). Here's a simple and efficient function to do this:\n\n```python\ndef is_palindrome(text):\n    \"\"\"\n    Check if the given string is a palindrome.\n    \n    Args:\n        text (str): The string to check\n        \n    Returns:\n        bool: True if the string is a palindrome, False otherwise\n    \"\"\"\n    # Convert to lowercase and remove non-alphanumeric characters\n    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())\n    \n    # Compare the string with its reverse\n    return cleaned_text == cleaned_text[::-1]\n```\n\nHere's how you can use this function:\n\n```python\n# Test cases\nprint(is_palindrome(\"racecar\"))               # True\nprint(is_palindrome(\"A man, a plan, a canal: Panama\"))  # True\nprint(is_palindrome(\"hello\"))                 # False\nprint(is_palindrome(\"Madam, I'm Adam\"))       # True\n```\n\nThis function works by:\n1. Converting the input string to lowercase\n2. Removing all non-alphanumeric characters (spaces, punctuation, etc.)\n3. Checking if the cleaned string is equal to its reverse\n\nIs there anything specific about this function you'd like me to explain or modify?"},
                {"role": "user", "content": "That looks good, but I'm concerned about efficiency. Is the string slicing with [::-1] efficient for very long strings? Could you provide an alternative approach?"},
                {"role": "assistant", "content": "You raise a great point about efficiency. While the string slicing approach (`[::-1]`) is clean and readable, it does create a full copy of the string, which could be inefficient for extremely long strings due to the additional memory allocation.\n\nHere's an alternative approach using a two-pointer technique that avoids creating a copy of the string and stops as soon as a mismatch is found:\n\n```python\ndef is_palindrome(text):\n    \"\"\"\n    Check if the given string is a palindrome using a two-pointer approach.\n    \n    Args:\n        text (str): The string to check\n        \n    Returns:\n        bool: True if the string is a palindrome, False otherwise\n    \"\"\"\n    # Convert to lowercase and remove non-alphanumeric characters\n    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())\n    \n    # Use two pointers approach\n    left, right = 0, len(cleaned_text) - 1\n    \n    while left < right:\n        if cleaned_text[left] != cleaned_text[right]:\n            return False\n        left += 1\n        right -= 1\n    \n    return True\n```\n\nBenefits of this approach:\n\n1. **Space Efficiency**: No additional string copy is created during the palindrome check\n2. **Early Termination**: Returns False immediately when a mismatch is found\n3. **Time Complexity**: Still O(n) but with less constant overhead\n\nComparison of both approaches:\n\n| Approach | Time Complexity | Space Complexity | Pros | Cons |\n|----------|----------------|------------------|------|------|\n| Slicing | O(n) | O(n) | Simple, Pythonic | Creates copy of string |\n| Two-pointer | O(n) | O(1) for comparison | Memory efficient, early exit | Slightly more code |\n\nThe cleaned_text creation is still O(n) in space complexity for both approaches, but the two-pointer method avoids the additional string copy for the reverse comparison.\n\nFor most practical applications with strings of reasonable length, both implementations will perform similarly. The two-pointer approach becomes advantageous when dealing with extremely long strings or in memory-constrained environments.\n\nWould you like me to provide any benchmarking code to compare the performance of these two approaches?"},
                {"role": "user", "content": "Yes, could you write a simple benchmark to compare both approaches?"},
                {"role": "assistant", "content": "Absolutely! Here's a benchmark script that compares both palindrome-checking approaches with strings of various lengths. This will give us a clear idea of the performance differences:\n\n```python\nimport time\nimport random\nimport string\nimport matplotlib.pyplot as plt\nfrom functools import wraps\n\n# Palindrome functions to compare\ndef is_palindrome_slice(text):\n    \"\"\"Palindrome check using string slicing.\"\"\"\n    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())\n    return cleaned_text == cleaned_text[::-1]\n\ndef is_palindrome_two_pointer(text):\n    \"\"\"Palindrome check using two-pointer technique.\"\"\"\n    cleaned_text = ''.join(char.lower() for char in text if char.isalnum())\n    \n    left, right = 0, len(cleaned_text) - 1\n    while left < right:\n        if cleaned_text[left] != cleaned_text[right]:\n            return False\n        left += 1\n        right -= 1\n    return True\n\n# Timing decorator\ndef time_it(func):\n    @wraps(func)\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n        result = func(*args, **kwargs)\n        end = time.perf_counter()\n        return end - start, result\n    return wrapper\n\n# Test function\n@time_it\ndef test_palindrome(func, text):\n    return func(text)\n\n# Generate test strings of different sizes\ndef generate_palindrome(length):\n    \"\"\"Generate a palindrome of specified length.\"\"\"\n    if length <= 0:\n        return \"\"\n    \n    # Generate half the palindrome (plus middle char if odd length)\n    half_length = length // 2\n    chars = random.choices(string.ascii_lowercase, k=half_length)\n    \n    # Create palindrome\n    if length % 2 == 0:  # Even length\n        return ''.join(chars + list(reversed(chars)))\n    else:  # Odd length\n        middle = random.choice(string.ascii_lowercase)\n        return ''.join(chars + [middle] + list(reversed(chars)))\n\n# Benchmark settings\nsizes = [100, 1000, 10000, 50000, 100000, 500000]\n# Add some punctuation and spaces to make it realistic\ndef add_noise(text, noise_ratio=0.2):\n    \"\"\"Add spaces and punctuation to make the text more realistic.\"\"\"\n    noise_chars = ' ,.!?;:\"\\'-_'\n    result = list(text)\n    \n    # Determine number of noise characters to insert\n    noise_count = int(len(text) * noise_ratio)\n    \n    # Insert noise at random positions\n    for _ in range(noise_count):\n        pos = random.randint(0, len(result))\n        noise = random.choice(noise_chars)\n        result.insert(pos, noise)\n        \n    # Randomly change case\n    for i in range(len(result)):\n        if result[i].isalpha() and random.random() < 0.3:  # 30% chance to change case\n            result[i] = result[i].upper()\n            \n    return ''.join(result)\n\n# Run benchmark\nresults = {'slice': [], 'two_pointer': []}\n\nprint(\"Running benchmark...\")\nprint(f\"{'Size':<10} {'Slice (s)':<15} {'Two-Pointer (s)':<15} {'Difference':<10}\")\nprint(\"-\" * 50)\n\nfor size in sizes:\n    # Generate a palindrome and add noise\n    palindrome = generate_palindrome(size)\n    test_text = add_noise(palindrome)\n    \n    # Run tests\n    slice_time, _ = test_palindrome(is_palindrome_slice, test_text)\n    two_pointer_time, _ = test_palindrome(is_palindrome_two_pointer, test_text)\n    \n    # Store results\n    results['slice'].append(slice_time)\n    results['two_pointer'].append(two_pointer_time)\n    \n    # Calculate difference\n    diff = ((slice_time - two_pointer_time) / slice_time) * 100 if slice_time > 0 else 0\n    \n    # Print results\n    print(f\"{size:<10} {slice_time:<15.6f} {two_pointer_time:<15.6f} {diff:+.2f}%\")\n\n# Plot results\nplt.figure(figsize=(10, 6))\nplt.plot(sizes, results['slice'], marker='o', label='String Slice Method')\nplt.plot(sizes, results['two_pointer'], marker='s', label='Two-Pointer Method')\nplt.xlabel('String Length')\nplt.ylabel('Time (seconds)')\nplt.title('Palindrome Check Performance')\nplt.legend()\nplt.grid(True)\nplt.xscale('log')\nplt.yscale('log')\nplt.tight_layout()\n\n# Add annotations for percentage difference at largest size\nmax_idx = -1\nslice_time = results['slice'][max_idx]\ntwo_pointer_time = results['two_pointer'][max_idx]\ndiff_pct = ((slice_time - two_pointer_time) / slice_time) * 100\nplt.annotate(f\"{diff_pct:.1f}% faster\", \n             xy=(sizes[max_idx], two_pointer_time),\n             xytext=(sizes[max_idx]*0.7, two_pointer_time*0.7),\n             arrowprops=dict(arrowstyle=\"->\", connectionstyle=\"arc3,rad=.2\"))\n\nplt.savefig('palindrome_benchmark.png')\nplt.show()\n\nprint(\"\\nBenchmark completed. Results indicate that:\")\nif sum(results['slice']) > sum(results['two_pointer']):\n    print(f\"- The two-pointer method is generally faster, with up to \"\n          f\"{((results['slice'][-1] - results['two_pointer'][-1]) / results['slice'][-1] * 100):.1f}% \"\n          f\"improvement for {sizes[-1]}-character strings.\")\nelse:\n    print(f\"- The slicing method is generally faster in this specific benchmark.\")\nprint(\"- The performance difference becomes more pronounced with larger strings.\")\nprint(\"- Both methods share the same string cleaning overhead, which dominates for small strings.\")\n```\n\nThis benchmark code:\n\n1. Implements both palindrome-checking methods\n2. Creates test palindromes of various sizes (with added noise like spaces and punctuation)\n3. Times each method for each string size\n4. Outputs results in a table and generates a plot for visual comparison\n\n### Expected Results\n\nYou should see that for small strings, the difference is negligible. As string size increases, the two-pointer method should show better performance due to:\n\n1. No need to create a full reversed copy of the string\n2. Potential early termination when a non-palindrome is detected\n\nThe exact percentage improvement will depend on your system, but I would expect anywhere from 5-30% better performance for very large strings with the two-pointer method.\n\n### Notes\n\n1. This benchmark requires `matplotlib` for plotting. If you don't have it installed, you can run `pip install matplotlib` first, or you can remove the plotting section.\n\n2. The cleaning step (`''.join(char.lower() for char in text if char.isalnum())`) is identical in both functions and usually dominates the runtime for both approaches. In practice, this means the actual difference between the two algorithms might be less significant than the theoretical difference.\n\nWould you like me to adjust the benchmark in any way, such as focusing on different string sizes or testing non-palindromes as well?"}
            ]
        }
    ]
    
    dataset = []
    for i in range(min(n, len(conversations))):
        dataset.append({
            "id": str(uuid.uuid4()),
            "messages": conversations[i]["messages"]
        })
    
    return dataset

# Main application
def main():
    st.markdown("<h1 class='main-header'>Foundation Model Fine-Tuning Dataset Guide</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    This guide demonstrates how to prepare datasets for fine-tuning foundation models. 
    Properly structured data is essential for successful model training and achieving desired outcomes.
    """)
    
    # Dataset Requirements Section
    st.markdown("<h2 class='sub-header' id='dataset-requirements'>Dataset Requirements</h2>", unsafe_allow_html=True)
    
    with st.expander("Dataset Structure & Format", expanded=True):
        st.markdown("""
        ### JSONL Format
        
        Most foundation model providers require training data in JSONL (JSON Lines) format, where each line is a valid JSON object representing one example.
        """)
        
        st.markdown("<div class='data-schema-container'>", unsafe_allow_html=True)
        st.code("""
# Each line is a separate JSON object
{"prompt": "What is AWS?", "completion": "Amazon Web Services (AWS) is a cloud computing platform..."}
{"prompt": "How to create an S3 bucket", "completion": "To create an S3 bucket, follow these steps..."}
""", language="json")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("### Size Requirements")
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.markdown("""
        Dataset size requirements depend on the task and model:
        
        - **Minimum**: Typically 100-1,000 examples
        - **Recommended**: 1,000-10,000 examples for most use cases
        - **High Performance**: 10,000+ examples for complex tasks
        
        *Quality is more important than quantity—smaller, high-quality datasets often outperform larger, lower-quality ones.*
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with st.expander("Data Quality Considerations"):
        st.markdown("<div class='requirement-item'>", unsafe_allow_html=True)
        st.markdown("""
        #### Diversity & Balance
        
        - Include diverse examples covering the range of inputs your model will encounter
        - Balance different categories or types of examples
        - Avoid repetitive patterns that could cause overfitting
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='requirement-item'>", unsafe_allow_html=True)
        st.markdown("""
        #### Consistency
        
        - Use consistent formatting and style across examples
        - For instruction tuning, use consistent instruction patterns
        - Apply consistent terminologies and conventions
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='requirement-item'>", unsafe_allow_html=True)
        st.markdown("""
        #### Representativeness
        
        - Examples should represent real-world use cases
        - Include edge cases and exceptions the model should handle
        - Match the distribution of inputs the model will receive in production
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='warning-box'>", unsafe_allow_html=True)
        st.markdown("""
        ⚠️ **Common Pitfalls to Avoid:**
        
        - Duplicate examples that can bias the model
        - Personal or sensitive information in training data
        - Copyright-protected material without permission
        - Inconsistent response styles or formats
        - Too few examples of important edge cases
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Example Datasets Section
    st.markdown("<h2 class='sub-header' id='example-datasets'>Example Datasets</h2>", unsafe_allow_html=True)
    
    dataset_tabs = st.tabs([
        "Text Classification", 
        "Question Answering", 
        "Summarization", 
        "Instruction Tuning", 
        "Chat Format"
    ])
    
    with dataset_tabs[0]:
        st.markdown("<div class='section-header'>Text Classification</div>", unsafe_allow_html=True)
        st.markdown("""
        Text classification involves categorizing text into predefined classes. This format is useful for sentiment analysis, topic categorization, intent classification, and more.
        """)
        
        classification_data = generate_text_classification_dataset(5)
        
        st.json(classification_data, expanded=False)
        
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Best Practices for Classification Datasets:**
        
        1. Maintain class balance when possible (or intentionally weight classes)
        2. Include ambiguous examples to improve model robustness
        3. Consider hierarchical labels for complex classification tasks
        4. Define clear boundaries between potentially overlapping classes
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display the first few examples in table format
        df = pd.DataFrame(classification_data)
        st.dataframe(df[['text', 'label']], use_container_width=True)
    
    with dataset_tabs[1]:
        st.markdown("<div class='section-header'>Question Answering</div>", unsafe_allow_html=True)
        st.markdown("""
        Question answering datasets pair questions with appropriate answers. This format is ideal for training models to provide specific answers to user queries.
        """)
        
        qa_data = generate_qa_dataset(5)
        
        st.json(qa_data, expanded=False)
        
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Best Practices for QA Datasets:**
        
        1. Include diverse question types (what, why, how, etc.)
        2. Provide answers with appropriate level of detail
        3. Consider including "I don't know" examples for questions outside scope
        4. Include questions requiring different types of reasoning
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display in two columns
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Sample Questions")
            for item in qa_data[:3]:
                st.markdown(f"**Q:** {item['question']}")
        
        with col2:
            st.markdown("#### Sample Answers")
            for item in qa_data[:3]:
                st.markdown(f"**A:** {item['answer'][:100]}...")
    
    with dataset_tabs[2]:
        st.markdown("<div class='section-header'>Summarization</div>", unsafe_allow_html=True)
        st.markdown("""
        Summarization datasets pair longer texts with their shorter summaries. This format trains models to extract key information and create concise summaries.
        """)
        
        summarization_data = generate_summarization_dataset(3)
        
        st.json(summarization_data[0], expanded=False)
        
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Best Practices for Summarization Datasets:**
        
        1. Ensure summaries capture key information from the original text
        2. Maintain consistent summary length and style
        3. Include diverse document types and subjects
        4. Be aware of potential biases in summary creation
        5. Verify summaries don't contain information not present in original text
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show one example with article and summary
        for i, item in enumerate(summarization_data[:2]):
            with st.expander(f"Example {i+1}: Article & Summary", expanded=(i==0)):
                st.markdown("#### Original Article")
                st.markdown(item["article"])
                st.markdown("#### Summary")
                st.markdown(f"<div style='background-color: #EFF6FF; padding: 10px; border-radius: 5px;'>{item['summary']}</div>", unsafe_allow_html=True)
    
    with dataset_tabs[3]:
        st.markdown("<div class='section-header'>Instruction Tuning</div>", unsafe_allow_html=True)
        st.markdown("""
        Instruction tuning uses a specific format with instructions, optional input context, and desired outputs. This approach is powerful for creating models that follow specific directions.
        """)
        
        instruction_data = generate_instruction_dataset(3)
        
        st.json(instruction_data[0], expanded=False)
        
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Best Practices for Instruction Datasets:**
        
        1. Use clear, specific instructions that match your intended use case
        2. Include diverse instruction types (create, explain, analyze, etc.)
        3. Make outputs comprehensive and high-quality
        4. Keep consistent instruction style and formatting
        5. Include examples with empty input fields for instructions that don't need context
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show sample instruction-output pairs
        for i, item in enumerate(instruction_data[:1]):
            with st.expander(f"Example: {item['instruction']}", expanded=True):
                st.markdown("#### Instruction")
                st.markdown(f"<div style='background-color: #E0F2FE; padding: 10px; border-radius: 5px;'>{item['instruction']}</div>", unsafe_allow_html=True)
                
                st.markdown("#### Input")
                st.markdown(f"<div style='background-color: #F0FDF4; padding: 10px; border-radius: 5px;'>{item['input'] if item['input'] else '(No input provided)'}</div>", unsafe_allow_html=True)
                
                st.markdown("#### Output")
                st.markdown(f"<div style='background-color: #F1F5F9; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;'>{item['output'][:500]}{'...' if len(item['output']) > 500 else ''}</div>", unsafe_allow_html=True)
                    
    with dataset_tabs[4]:
        st.markdown("<div class='section-header'>Chat Format</div>", unsafe_allow_html=True)
        st.markdown("""
        Chat format datasets contain multi-turn conversations with system messages, user inputs, and assistant responses. This format is ideal for training conversational models.
        """)
        
        chat_data = generate_chat_dataset(2)
        
        st.json({"messages": chat_data[0]["messages"][:3]}, expanded=False)
        
        st.markdown("<div class='success-box'>", unsafe_allow_html=True)
        st.markdown("""
        **Best Practices for Chat Datasets:**
        
        1. Include system messages to set context and behavior
        2. Create multi-turn conversations with natural flow
        3. Represent different conversation types and user needs
        4. Show appropriate handling of challenging or ambiguous queries
        5. Demonstrate consistent assistant persona and tone
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Display a sample conversation
        with st.expander("Sample Conversation", expanded=True):
            for message in chat_data[0]["messages"]:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    st.markdown(f"<div style='background-color: #F0F4F8; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>System:</strong> {content}</div>", unsafe_allow_html=True)
                elif role == "user":
                    st.markdown(f"<div style='background-color: #E0F2FE; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>User:</strong> {content}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='background-color: #F0FDF4; padding: 10px; border-radius: 5px; margin-bottom: 10px;'><strong>Assistant:</strong> {content[:200]}{'...' if len(content) > 200 else ''}</div>", unsafe_allow_html=True)
    
    # Data Preparation Section
    st.markdown("<h2 class='sub-header' id='data-preparation'>Data Preparation</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Preparation Process</div>", unsafe_allow_html=True)
        st.markdown("""
        1. **Data Collection**
           - Gather or create examples matching your use case
           - Ensure proper licensing and permissions
        
        2. **Data Cleaning**
           - Remove duplicates and inconsistencies
           - Check for and handle missing values
           - Normalize formatting and structure
        
        3. **Data Validation**
           - Verify JSON/JSONL format correctness
           - Check for required fields in each example
           - Validate against provider-specific requirements
        
        4. **Train/Test Split**
           - Divide into training (~80-90%) and validation (10-20%) sets
           - Ensure representative distribution in both sets
        """)
    
    with col2:
        st.markdown("<div class='section-header'>Dataset Partitioning</div>", unsafe_allow_html=True)
        
        st.markdown("""
        Typical dataset split for fine-tuning:
        """)
        
        # Create placeholder data for pie chart
        labels = 'Training', 'Validation', 'Test (Optional)'
        sizes = [80, 15, 5]
        colors = ['#3B82F6', '#10B981', '#F59E0B']
        explode = (0, 0.1, 0.1)
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
              shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        st.pyplot(fig)
        
        st.markdown("""
        - **Training set**: Used to train the model
        - **Validation set**: Used to evaluate during training
        - **Test set (optional)**: For final evaluation before deployment
        """)
    
    # Upload Process Section
    with st.expander("Dataset Upload Process", expanded=False):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Upload Process Flow")
            
            st.markdown("""
            ```mermaid
            graph TD
                A[Prepare JSONL Files] --> B[Create S3 Buckets]
                B --> C[Upload Files to S3]
                C --> D[Configure IAM Permissions]
                D --> E[Reference Data in Fine-tuning Job]
                E --> F[Monitor Training Progress]
            ```
            """)
        
        st.markdown("""
        1. **Create S3 buckets**
           - One bucket for training data
           - One bucket for output model artifacts
        
        2. **Upload data files**
           - Upload your training.jsonl to the training bucket
           - Upload validation.jsonl (if used) to the training bucket
        
        3. **Configure permissions**
           - Grant the fine-tuning service access to your S3 buckets
           - Set appropriate IAM policies
        
        4. **Initiate fine-tuning job**
           - Specify S3 URIs for your data
           - Select model, hyperparameters and training configuration
        """)
    
    # Best Practices Section
    st.markdown("<h2 class='sub-header' id='best-practices'>Best Practices</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='section-header'>Technical Best Practices</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Validate JSONL format** before uploading
        - **Start small** and iterate with model performance
        - **Use consistent formatting** across all examples
        - **Include special tokens** properly if required by the model
        - **Consider token limits** of your target model
        - **Test with synthetic examples** to confirm behaviors
        """)
    
    with col2:
        st.markdown("<div class='section-header'>Quality Best Practices</div>", unsafe_allow_html=True)
        st.markdown("""
        - **Human review** of training examples before submission
        - **Avoid overfitting** with diverse examples
        - **Balance** different use cases and scenarios
        - **Include edge cases** the model should handle well
        - **Test for biases** that might be amplified during training
        - **Document dataset characteristics** for future reference
        """)
    
    # Tools Section
    with st.expander("Helpful Tools", expanded=False):
        st.markdown("""
        ### Dataset Preparation Tools
        
        #### 1. Data Validation
        ```python
        import json
        import jsonschema
        
        # Define schema
        schema = {
            "type": "object",
            "required": ["prompt", "completion"],
            "properties": {
                "prompt": {"type": "string"},
                "completion": {"type": "string"}
            }
        }
        
        # Validate JSONL file
        def validate_jsonl(filename, schema):
            errors = []
            with open(filename, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        jsonschema.validate(instance=data, schema=schema)
                    except Exception as e:
                        errors.append(f"Error at line {i+1}: {str(e)}")
            return errors
        ```
        
        #### 2. Token Counting
        ```python
        import tiktoken
        
        def count_tokens(text, model="gpt-3.5-turbo"):
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        
        def analyze_dataset_tokens(filename):
            token_counts = {"prompt": [], "completion": []}
            with open(filename, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    token_counts["prompt"].append(count_tokens(data["prompt"]))
                    token_counts["completion"].append(count_tokens(data["completion"]))
            
            return {
                "prompt_avg": sum(token_counts["prompt"]) / len(token_counts["prompt"]),
                "completion_avg": sum(token_counts["completion"]) / len(token_counts["completion"]),
                "prompt_max": max(token_counts["prompt"]),
                "completion_max": max(token_counts["completion"])
            }
        ```
        
        #### 3. S3 Upload Helper
        ```python
        import boto3
        
        def upload_to_s3(file_path, bucket, object_name=None):
            if object_name is None:
                object_name = file_path
                
            s3_client = boto3.client('s3')
            try:
                s3_client.upload_file(file_path, bucket, object_name)
                return True
            except Exception as e:
                print(f"Error uploading file: {str(e)}")
                return False
        ```
        """)
    
    # # Download Example Datasets
    # st.markdown("<h2 class='sub-header'>Download Example Datasets</h2>", unsafe_allow_html=True)
    
    # col1, col2, col3, col4, col5 = st.columns(5)
    
    # with col1:
    #     st.markdown("<a href='#' class='download-button'>Classification Dataset</a>", unsafe_allow_html=True)
    
    # with col2:
    #     st.markdown("<a href='#' class='download-button'>QA Dataset</a>", unsafe_allow_html=True)
    
    # with col3:
    #     st.markdown("<a href='#' class='download-button'>Summarization Dataset</a>", unsafe_allow_html=True)
    
    # with col4:
    #     st.markdown("<a href='#' class='download-button'>Instruction Dataset</a>", unsafe_allow_html=True)
    
    # with col5:
    #     st.markdown("<a href='#' class='download-button'>Chat Dataset</a>", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("Created for foundation model fine-tuning - For educational purposes only")

if __name__ == "__main__":
    main()
