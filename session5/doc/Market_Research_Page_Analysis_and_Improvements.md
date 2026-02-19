# Market Research Page - Analysis & Improvement Suggestions

## Executive Summary

The Market Research page is a well-architected Streamlit application that uses AWS Bedrock and RAG (Retrieval Augmented Generation) to analyze shoe industry market research documents. The code demonstrates good software engineering practices with clear separation of concerns, type hints, and comprehensive functionality.

**Overall Assessment: 8.5/10**

---

## Current Strengths

### 1. Architecture & Code Quality
- ✅ Clean class-based architecture with clear separation of concerns
- ✅ Type hints and docstrings throughout
- ✅ Proper error handling and logging with structlog
- ✅ Session state management
- ✅ Modern CSS styling with custom themes

### 2. Functionality
- ✅ Multi-document upload support (PDF, DOCX, TXT)
- ✅ RAG implementation with FAISS vector store
- ✅ Comprehensive market analysis across 4 tabs
- ✅ Multiple AI model selection (15+ models)
- ✅ Interactive visualizations with Plotly
- ✅ Financial projections and risk assessment

### 3. User Experience
- ✅ Modern, gradient-based UI design
- ✅ Responsive layout with columns
- ✅ Progress indicators and spinners
- ✅ Expandable sections for details
- ✅ Clear navigation with tabs

---

## Identified Issues & Improvements

### CRITICAL ISSUES

#### 1. **Plotly Chart Key Conflicts** ⚠️ HIGH PRIORITY
**Issue:** Multiple Plotly charts use duplicate or missing keys, causing Streamlit errors.

**Current Problems:**
- Line 1031: `key="plotly1"` 
- Line 1073: `key="plotly2"`
- Line 1155: `key="plotly_just_{idx}"` (good - uses index)
- Line 1217: `key="plotly_financial"` 
- Line 1307: `key="plotly7"`
- Line 1456: `key="plotly8"`
- Line 1509: `key="plotly9"`
- Line 1571: `key="plotly5"` (DUPLICATE - already used elsewhere)
- Line 1625: `key="plotly4"`

**Solution:**
```python
# Use descriptive, unique keys for all charts
key="chart_market_growth"
key="chart_market_share_treemap"
key="chart_justification_demand"
key="chart_financial_projection"
key="chart_competitive_positioning"
key="chart_material_composition"
key="chart_performance_radar"
key="chart_sustainability_comparison"
key="chart_development_timeline"
```

#### 2. **Unsafe Numeric Parsing** ⚠️ MEDIUM PRIORITY
**Issue:** Growth rate parsing at line 1024 could fail with unexpected formats.

**Current Code:**
```python
growth_match = re.search(r'(\d+(?:\.\d+)?)', growth_rate_str)
if growth_match:
    try:
        growth_value = float(growth_match.group(1))
```

**Improvement:**
```python
def safe_extract_numeric(text: str, default: float = 0.0) -> float:
    """Safely extract numeric value from text"""
    if not text or text == "Not specified":
        return default
    
    # Try multiple patterns
    patterns = [
        r'(\d+(?:\.\d+)?)\s*%',  # "5.5%"
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',  # "5-7"
        r'(\d+(?:\.\d+)?)'  # "5.5"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                # For range, take average
                if match.lastindex == 2:
                    return (float(match.group(1)) + float(match.group(2))) / 2
                return float(match.group(1))
            except (ValueError, AttributeError):
                continue
    
    return default
```

---

### FUNCTIONALITY IMPROVEMENTS

#### 3. **Enhanced Document Processing**
**Current Limitation:** Basic text splitting without semantic awareness.

**Improvements:**
```python
class DocumentProcessor:
    def __init__(self, bedrock_service: BedrockService):
        self.bedrock_service = bedrock_service
        
        # Add semantic-aware splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,  # Increased from 1000
            chunk_overlap=200,  # Increased from 100 for better context
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraph breaks
                "\n",      # Line breaks
                ". ",      # Sentence breaks
                " ",       # Word breaks
                ""
            ],
            length_function=len,
        )
    
    def process_files(self, files: List) -> Tuple[Optional[FAISS], List[Document]]:
        """Enhanced with metadata extraction"""
        all_documents = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                try:
                    temp_path = Path(temp_dir) / file.name
                    temp_path.write_bytes(file.getvalue())
                    
                    documents = self._load_document(str(temp_path))
                    
                    # Add metadata
                    for doc in documents:
                        doc.metadata.update({
                            'source_file': file.name,
                            'file_type': Path(file.name).suffix,
                            'upload_time': datetime.now().isoformat()
                        })
                    
                    all_documents.extend(documents)
                    
                except Exception as e:
                    logger.error(f"Error processing {file.name}: {str(e)}")
                    st.error(f"Error loading {file.name}: {str(e)}")
        
        # ... rest of the method
```

#### 4. **Caching for Performance**
**Issue:** Repeated analysis calls regenerate the same results.

**Solution:**
```python
import hashlib
from functools import lru_cache

class MarketAnalyzer:
    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag_engine = rag_engine
        self._cache = {}
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        return hashlib.md5(query.encode()).hexdigest()
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def analyze_document(_self) -> Dict[str, Any]:
        """Cached analysis to avoid redundant API calls"""
        cache_key = "full_analysis"
        
        if cache_key in _self._cache:
            return _self._cache[cache_key]
        
        analysis = {
            'opportunities': _self._extract_opportunities(),
            'market_insights': _self._extract_market_insights(),
            'customer_segments': _self._extract_customer_segments(),
            'competitive_landscape': _self._extract_competitive_landscape(),
            'feasibility_score': _self._calculate_feasibility_score()
        }
        
        _self._cache[cache_key] = analysis
        return analysis
```

#### 5. **Export Functionality** 🆕
**Missing Feature:** Users cannot export analysis results.

**Implementation:**
```python
def create_export_section(analysis: Dict[str, Any], recommendations: Dict[str, Any], specs: Dict[str, Any]):
    """Add export functionality"""
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export as JSON
        if st.button("📄 Export JSON", use_container_width=True):
            export_data = {
                'analysis': analysis,
                'recommendations': recommendations,
                'specifications': specs,
                'export_date': datetime.now().isoformat()
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"market_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        # Export as PDF report
        if st.button("📑 Export PDF", use_container_width=True):
            pdf_buffer = generate_pdf_report(analysis, recommendations, specs)
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name=f"market_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
    
    with col3:
        # Export as Excel
        if st.button("📊 Export Excel", use_container_width=True):
            excel_buffer = generate_excel_report(analysis, recommendations, specs)
            st.download_button(
                label="Download Excel",
                data=excel_buffer,
                file_name=f"market_data_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
```

#### 6. **Comparison Mode** 🆕
**Missing Feature:** Cannot compare multiple documents or scenarios.

**Implementation:**
```python
def create_comparison_mode():
    """Allow users to compare multiple analyses"""
    st.markdown("### 🔄 Comparison Mode")
    
    if 'comparison_analyses' not in st.session_state:
        st.session_state.comparison_analyses = []
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        scenario_name = st.text_input("Scenario Name", placeholder="e.g., Premium Market Entry")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save Scenario", use_container_width=True):
            if st.session_state.analysis_results and scenario_name:
                st.session_state.comparison_analyses.append({
                    'name': scenario_name,
                    'analysis': st.session_state.analysis_results,
                    'timestamp': datetime.now()
                })
                st.success(f"Saved: {scenario_name}")
    
    # Display comparison table
    if len(st.session_state.comparison_analyses) >= 2:
        comparison_data = []
        for scenario in st.session_state.comparison_analyses:
            comparison_data.append({
                'Scenario': scenario['name'],
                'Feasibility': f"{scenario['analysis']['feasibility_score']}%",
                'Opportunities': len(scenario['analysis']['opportunities']),
                'Market Size': scenario['analysis']['market_insights'].get('market_size', 'N/A'),
                'Saved': scenario['timestamp'].strftime('%Y-%m-%d %H:%M')
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        # Side-by-side comparison charts
        fig = go.Figure()
        for scenario in st.session_state.comparison_analyses:
            fig.add_trace(go.Bar(
                name=scenario['name'],
                x=['Feasibility', 'Opportunities', 'Segments'],
                y=[
                    scenario['analysis']['feasibility_score'],
                    len(scenario['analysis']['opportunities']) * 10,
                    len(scenario['analysis']['customer_segments']) * 20
                ]
            ))
        
        fig.update_layout(
            title='Scenario Comparison',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_scenario_comparison")
```

---

### UI/UX IMPROVEMENTS

#### 7. **Progress Tracking**
**Enhancement:** Show analysis progress with detailed steps.

**Implementation:**
```python
def analyze_with_progress(analyzer: MarketAnalyzer):
    """Enhanced analysis with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    steps = [
        ("Extracting opportunities", 20),
        ("Analyzing market insights", 40),
        ("Identifying customer segments", 60),
        ("Evaluating competition", 80),
        ("Calculating feasibility", 100)
    ]
    
    analysis = {}
    
    for step_name, progress in steps:
        status_text.text(f"🔄 {step_name}...")
        progress_bar.progress(progress)
        
        if "opportunities" not in analysis:
            analysis['opportunities'] = analyzer._extract_opportunities()
        elif "market_insights" not in analysis:
            analysis['market_insights'] = analyzer._extract_market_insights()
        elif "customer_segments" not in analysis:
            analysis['customer_segments'] = analyzer._extract_customer_segments()
        elif "competitive_landscape" not in analysis:
            analysis['competitive_landscape'] = analyzer._extract_competitive_landscape()
        else:
            analysis['feasibility_score'] = analyzer._calculate_feasibility_score()
    
    status_text.text("✅ Analysis complete!")
    progress_bar.progress(100)
    
    return analysis
```

#### 8. **Interactive Filters**
**Enhancement:** Allow users to filter and drill down into data.

**Implementation:**
```python
def create_interactive_filters():
    """Add filtering capabilities"""
    st.sidebar.markdown("### 🔍 Filters")
    
    # Feasibility threshold
    min_feasibility = st.sidebar.slider(
        "Minimum Feasibility Score",
        min_value=0,
        max_value=100,
        value=50,
        help="Filter opportunities by feasibility threshold"
    )
    
    # Market segment filter
    segment_options = ["All", "Gen Z", "Millennials", "Gen X", "Baby Boomers"]
    selected_segments = st.sidebar.multiselect(
        "Target Segments",
        options=segment_options,
        default=["All"]
    )
    
    # Price range filter
    price_range = st.sidebar.slider(
        "Price Range ($)",
        min_value=0,
        max_value=500,
        value=(50, 200),
        help="Filter by target price range"
    )
    
    return {
        'min_feasibility': min_feasibility,
        'segments': selected_segments,
        'price_range': price_range
    }
```

#### 9. **Tooltips and Help**
**Enhancement:** Add contextual help throughout the interface.

**Implementation:**
```python
def add_contextual_help():
    """Add help tooltips and information"""
    
    # Add help icon with popover
    st.markdown("""
    <style>
    .help-icon {
        display: inline-block;
        width: 20px;
        height: 20px;
        background-color: #3b82f6;
        color: white;
        border-radius: 50%;
        text-align: center;
        line-height: 20px;
        font-size: 12px;
        cursor: help;
        margin-left: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Usage example
    st.markdown("""
    Feasibility Score <span class="help-icon" title="Score based on market size, growth rate, competition, and opportunities">?</span>
    """, unsafe_allow_html=True)
```

#### 10. **Responsive Mobile Design**
**Enhancement:** Improve mobile experience.

**Implementation:**
```python
def load_custom_css():
    """Enhanced CSS with mobile responsiveness"""
    st.markdown("""
    <style>
    /* Existing styles... */
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 0.25rem 0;
        }
        
        .analysis-card {
            padding: 1rem;
        }
        
        /* Stack columns on mobile */
        .stColumns {
            flex-direction: column;
        }
    }
    
    /* Tablet */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)
```

---

### ADVANCED FEATURES

#### 11. **AI Chat Assistant** 🆕
**New Feature:** Interactive Q&A about the analysis.

**Implementation:**
```python
def create_chat_assistant(rag_engine: RAGQueryEngine):
    """Add interactive chat for document Q&A"""
    st.markdown("### 💬 Ask Questions About Your Analysis")
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the market research..."):
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                system_prompt = "You are a market research expert. Answer questions based on the provided context."
                response = rag_engine.query(prompt, system_prompt)
                
                if response:
                    st.markdown(response)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                else:
                    st.error("Failed to generate response")
```

#### 12. **Automated Insights** 🆕
**New Feature:** AI-generated key insights and recommendations.

**Implementation:**
```python
def generate_automated_insights(analysis: Dict[str, Any]) -> List[str]:
    """Generate automated insights from analysis"""
    insights = []
    
    # Feasibility insight
    feasibility = analysis['feasibility_score']
    if feasibility > 80:
        insights.append(f"🎯 Strong Market Opportunity: With a feasibility score of {feasibility}%, this market shows excellent potential for entry.")
    elif feasibility > 60:
        insights.append(f"⚠️ Moderate Opportunity: Feasibility score of {feasibility}% suggests careful positioning is needed.")
    else:
        insights.append(f"🔴 High Risk: Feasibility score of {feasibility}% indicates significant challenges.")
    
    # Growth insight
    growth_rate = analysis['market_insights'].get('growth_rate', '')
    if 'growth' in growth_rate.lower():
        insights.append(f"📈 Market Growth: {growth_rate} indicates expanding market opportunities.")
    
    # Competition insight
    num_competitors = len(analysis['competitive_landscape']['major_players'])
    if num_competitors > 5:
        insights.append(f"🏆 Competitive Market: {num_competitors} major players identified - differentiation is critical.")
    
    # Opportunity insight
    num_opportunities = len(analysis['opportunities'])
    if num_opportunities >= 5:
        insights.append(f"💡 Rich Opportunity Landscape: {num_opportunities} distinct opportunities identified for market entry.")
    
    return insights

def display_automated_insights(analysis: Dict[str, Any]):
    """Display automated insights prominently"""
    insights = generate_automated_insights(analysis)
    
    st.markdown("### 🤖 AI-Generated Insights")
    
    for insight in insights:
        st.info(insight)
```

#### 13. **Sensitivity Analysis** 🆕
**New Feature:** Show how changes in assumptions affect outcomes.

**Implementation:**
```python
def create_sensitivity_analysis(base_feasibility: int):
    """Interactive sensitivity analysis"""
    st.markdown("### 📊 Sensitivity Analysis")
    
    st.markdown("Adjust key assumptions to see impact on feasibility:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        market_growth_adj = st.slider(
            "Market Growth Adjustment",
            min_value=-50,
            max_value=50,
            value=0,
            help="Adjust market growth rate assumption"
        )
        
        competition_adj = st.slider(
            "Competition Level",
            min_value=-30,
            max_value=30,
            value=0,
            help="Adjust competitive intensity"
        )
    
    with col2:
        cost_adj = st.slider(
            "Cost Structure",
            min_value=-40,
            max_value=40,
            value=0,
            help="Adjust cost assumptions"
        )
        
        demand_adj = st.slider(
            "Demand Strength",
            min_value=-30,
            max_value=30,
            value=0,
            help="Adjust demand assumptions"
        )
    
    # Calculate adjusted feasibility
    adjusted_feasibility = base_feasibility + (
        (market_growth_adj * 0.3) +
        (competition_adj * -0.2) +
        (cost_adj * -0.25) +
        (demand_adj * 0.25)
    )
    adjusted_feasibility = max(0, min(100, int(adjusted_feasibility)))
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Base Feasibility", f"{base_feasibility}%")
    
    with col2:
        delta = adjusted_feasibility - base_feasibility
        st.metric(
            "Adjusted Feasibility",
            f"{adjusted_feasibility}%",
            f"{delta:+d}%"
        )
    
    with col3:
        impact = "Positive" if delta > 0 else "Negative" if delta < 0 else "Neutral"
        st.metric("Impact", impact)
    
    # Tornado chart showing sensitivity
    factors = ['Market Growth', 'Competition', 'Costs', 'Demand']
    impacts = [
        market_growth_adj * 0.3,
        competition_adj * -0.2,
        cost_adj * -0.25,
        demand_adj * 0.25
    ]
    
    fig = go.Figure()
    
    colors = ['#10b981' if x > 0 else '#ef4444' for x in impacts]
    
    fig.add_trace(go.Bar(
        y=factors,
        x=impacts,
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title='Sensitivity Impact on Feasibility',
        xaxis_title='Impact on Feasibility Score',
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True, key="chart_sensitivity")
```

---

### PERFORMANCE OPTIMIZATIONS

#### 14. **Lazy Loading**
**Optimization:** Load heavy components only when needed.

**Implementation:**
```python
def lazy_load_visualizations():
    """Load visualizations only when tab is active"""
    
    # Use st.empty() placeholders
    chart_placeholder = st.empty()
    
    # Load only when user scrolls to section
    if st.session_state.get('show_charts', False):
        with chart_placeholder.container():
            # Render expensive charts here
            pass
    else:
        with chart_placeholder.container():
            if st.button("📊 Load Visualizations"):
                st.session_state.show_charts = True
                st.rerun()
```

#### 15. **Batch API Calls**
**Optimization:** Reduce API calls by batching queries.

**Implementation:**
```python
class MarketAnalyzer:
    def analyze_document_batch(self) -> Dict[str, Any]:
        """Batch multiple queries into single API call"""
        
        combined_query = """
        Analyze the market research document and provide:
        1. Top 5 market opportunities
        2. Market size, growth rate, and key trends
        3. Customer segments with characteristics
        4. Competitive landscape analysis
        5. Overall feasibility assessment
        
        Format your response with clear section headers.
        """
        
        system_prompt = "You are a comprehensive market research analyst."
        
        response = self.rag_engine.query(combined_query, system_prompt)
        
        if response:
            # Parse the combined response
            analysis = self._parse_combined_response(response)
            return analysis
        
        # Fallback to individual queries
        return self._analyze_individual()
```

---

## Priority Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
1. ✅ Fix Plotly chart key conflicts
2. ✅ Improve numeric parsing safety
3. ✅ Add error boundaries for all API calls

### Phase 2: Core Enhancements (Week 2-3)
4. ✅ Add export functionality (JSON, PDF, Excel)
5. ✅ Implement caching for performance
6. ✅ Add progress tracking for analysis
7. ✅ Enhance document processing with metadata

### Phase 3: Advanced Features (Week 4-5)
8. ✅ Add comparison mode for scenarios
9. ✅ Implement AI chat assistant
10. ✅ Add automated insights generation
11. ✅ Create sensitivity analysis tool

### Phase 4: Polish & Optimization (Week 6)
12. ✅ Improve mobile responsiveness
13. ✅ Add contextual help and tooltips
14. ✅ Implement lazy loading
15. ✅ Optimize API call batching

---

## Specific Code Changes

### File: `session5/pages/02_Market_Research.py`

#### Change 1: Fix Chart Keys (Lines 1024-1625)
```python
# Replace all plotly chart keys with descriptive unique names:
key="chart_market_growth_gauge"           # Line 1031
key="chart_market_share_treemap"          # Line 1073
key="chart_justification_{idx}"           # Line 1155 (already good)
key="chart_financial_projection"          # Line 1217
key="chart_competitive_positioning"       # Line 1307
key="chart_material_composition_pie"      # Line 1456
key="chart_performance_radar"             # Line 1509
key="chart_sustainability_comparison_bar" # Line 1571
key="chart_development_timeline_gantt"    # Line 1625
```

#### Change 2: Add Safe Numeric Extraction (After line 35)
```python
def safe_extract_numeric(text: str, default: float = 0.0) -> float:
    """Safely extract numeric value from text with multiple pattern support"""
    if not text or text == "Not specified":
        return default
    
    patterns = [
        r'(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if match.lastindex == 2:
                    return (float(match.group(1)) + float(match.group(2))) / 2
                return float(match.group(1))
            except (ValueError, AttributeError):
                continue
    
    return default
```

#### Change 3: Add Export Section (After line 1700)
```python
def create_export_section(analysis: Dict[str, Any]):
    """Add export functionality to sidebar"""
    st.sidebar.markdown("### 📥 Export")
    
    if st.sidebar.button("📄 Export JSON", use_container_width=True):
        json_str = json.dumps(analysis, indent=2, default=str)
        st.sidebar.download_button(
            label="Download",
            data=json_str,
            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
```

---

## Testing Recommendations

### Unit Tests Needed
1. Test numeric extraction with various formats
2. Test document processing with corrupted files
3. Test RAG query with empty vectorstore
4. Test chart rendering with missing data

### Integration Tests
1. End-to-end document upload and analysis
2. Multi-document processing
3. Export functionality
4. Model switching

### Performance Tests
1. Large document processing (100+ pages)
2. Multiple concurrent users
3. API rate limiting handling
4. Memory usage with large vectorstores

---

## Conclusion

The Market Research page is well-built with solid architecture. The suggested improvements focus on:

1. **Reliability**: Fixing chart key conflicts and improving error handling
2. **Functionality**: Adding export, comparison, and chat features
3. **User Experience**: Better progress tracking, filters, and mobile support
4. **Performance**: Caching, lazy loading, and batch API calls

**Estimated Development Time**: 4-6 weeks for full implementation
**Priority**: Start with Phase 1 (critical fixes) immediately

The application has strong potential to become a comprehensive market research platform with these enhancements.
