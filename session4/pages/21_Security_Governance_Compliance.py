import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import utils.common as common
import utils.authenticate as authenticate
from utils.styles import load_css, custom_header

# Page configuration
st.set_page_config(
    page_title="AWS GenAI Security, Governance & Compliance",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

common.initialize_session_state()
load_css()

# ──────────────────────────────────────────────
# Data definitions
# ──────────────────────────────────────────────

SECURITY_SERVICES = {
    "AWS IAM": {
        "icon": "🔑",
        "tagline": "Identity & Access Management",
        "color": "#DD344C",
        "description": (
            "Controls who can access your GenAI resources and what actions they can perform. "
            "For generative AI workloads, IAM policies enforce least-privilege access to "
            "Bedrock models, SageMaker endpoints, training data in S3, and inference APIs."
        ),
        "genai_use_cases": [
            "Restrict which foundation models a team can invoke on Amazon Bedrock",
            "Scope SageMaker notebook access to specific training buckets",
            "Enforce MFA for any action that modifies model endpoints",
            "Use service-control policies (SCPs) to prevent unauthorized model deployment",
            "Tag-based access control for multi-tenant GenAI platforms",
        ],
        "key_features": [
            "Fine-grained resource policies",
            "Service-linked roles for Bedrock & SageMaker",
            "Permission boundaries for data-science teams",
            "Cross-account access for shared model registries",
        ],
        "best_practices": [
            "Apply least-privilege: grant only the Bedrock model IDs each role needs",
            "Use IAM Conditions to restrict API calls by IP, VPC, or time window",
            "Rotate credentials and prefer IAM Roles over long-lived keys",
            "Enable IAM Access Analyzer to find overly permissive policies",
        ],
    },
    "AWS KMS": {
        "icon": "🗝️",
        "tagline": "Key Management Service",
        "color": "#DD344C",
        "description": (
            "Manages encryption keys that protect data at rest and in transit across your "
            "GenAI pipeline — from training datasets in S3, to model artifacts, to inference "
            "request/response payloads."
        ),
        "genai_use_cases": [
            "Encrypt training data and model artifacts in S3 with customer-managed CMKs",
            "Encrypt SageMaker training volumes and endpoint storage",
            "Protect Amazon Bedrock custom-model imports with dedicated keys",
            "Encrypt vector database stores (e.g., OpenSearch, Aurora) used for RAG",
            "Audit every key usage event via CloudTrail integration",
        ],
        "key_features": [
            "Customer-managed keys (CMKs) with automatic rotation",
            "Envelope encryption for large datasets",
            "Key policies with grant-based delegation",
            "FIPS 140-2 Level 3 HSMs (CloudHSM integration)",
        ],
        "best_practices": [
            "Use separate CMKs for training data vs. model artifacts vs. inference logs",
            "Enable automatic key rotation (every 365 days)",
            "Restrict key usage to specific IAM principals and services",
            "Monitor key usage with CloudTrail and set alarms for anomalies",
        ],
    },
    "Amazon CloudWatch": {
        "icon": "📊",
        "tagline": "Monitoring & Observability",
        "color": "#FF9900",
        "description": (
            "Provides real-time monitoring, logging, and alerting for GenAI workloads. "
            "Tracks model invocation metrics, latency, error rates, token usage, and "
            "operational health of inference endpoints."
        ),
        "genai_use_cases": [
            "Monitor Bedrock model invocation counts, latency, and throttling",
            "Track SageMaker endpoint CPU/GPU utilization and auto-scaling events",
            "Log and alert on guardrail intervention rates",
            "Create dashboards for token consumption and cost tracking",
            "Detect anomalous usage patterns that may indicate prompt injection attacks",
        ],
        "key_features": [
            "Custom metrics and dashboards",
            "CloudWatch Logs Insights for querying inference logs",
            "Composite alarms for multi-signal alerting",
            "Anomaly detection with ML-powered baselines",
        ],
        "best_practices": [
            "Set alarms on Bedrock throttling and 4xx/5xx error rates",
            "Use Logs Insights to analyze guardrail block patterns",
            "Create cost-anomaly dashboards for token-based billing",
            "Enable detailed monitoring for SageMaker endpoints in production",
        ],
    },
    "AWS Audit Manager": {
        "icon": "📋",
        "tagline": "Continuous Audit & Evidence Collection",
        "color": "#3F8624",
        "description": (
            "Automates evidence collection to continuously audit your GenAI workloads "
            "against regulatory frameworks. Maps AWS resource configurations and activity "
            "logs to compliance controls for HIPAA, SOC 2, GDPR, NIST, and custom frameworks."
        ),
        "genai_use_cases": [
            "Automatically collect evidence for AI governance audits",
            "Map Bedrock guardrail configurations to compliance controls",
            "Track data lineage and model versioning for regulatory submissions",
            "Generate audit-ready reports for regulated industry reviews",
            "Assess compliance posture against the NIST AI Risk Management Framework",
        ],
        "key_features": [
            "Pre-built frameworks (HIPAA, SOC 2, GDPR, NIST 800-53)",
            "Custom framework builder for AI-specific controls",
            "Automated evidence collection from CloudTrail, Config, and Security Hub",
            "Assessment reports with delegation workflows",
        ],
        "best_practices": [
            "Create a custom framework mapping AI-specific controls (bias, explainability, data governance)",
            "Schedule recurring assessments aligned with model retraining cycles",
            "Delegate evidence review to domain experts (legal, compliance, ML engineering)",
            "Integrate with AWS Config rules for continuous compliance monitoring",
        ],
    },
    "AWS Security Hub": {
        "icon": "🛡️",
        "tagline": "Centralized Security Posture Management",
        "color": "#DD344C",
        "description": (
            "Aggregates security findings from across your AWS environment into a single "
            "pane of glass. For GenAI workloads, it consolidates findings from GuardDuty, "
            "IAM Access Analyzer, Macie, Inspector, and Config to surface misconfigurations "
            "and threats targeting your AI infrastructure."
        ),
        "genai_use_cases": [
            "Detect publicly accessible SageMaker notebooks or endpoints",
            "Identify S3 buckets with training data that lack encryption or logging",
            "Surface IAM findings for overly permissive Bedrock access policies",
            "Correlate GuardDuty threat findings with AI workload resources",
            "Track compliance scores against security standards (CIS, PCI-DSS, NIST)",
        ],
        "key_features": [
            "Automated security checks against best-practice standards",
            "Cross-account and cross-region aggregation",
            "Integration with 70+ AWS and partner services",
            "Custom insights and automated remediation actions",
        ],
        "best_practices": [
            "Enable the AWS Foundational Security Best Practices standard",
            "Create custom insights filtering for GenAI-related resources (Bedrock, SageMaker, S3)",
            "Set up automated remediation for critical findings via EventBridge + Lambda",
            "Review Security Hub scores weekly as part of AI governance reviews",
        ],
    },
}

INDUSTRY_FRAMEWORKS = {
    "Healthcare": {
        "icon": "🏥",
        "color": "#1E88E5",
        "regulations": [
            {"name": "HIPAA (updated Feb 2026)", "detail": "AI-specific risk analyses required for agentic AI systems accessing PHI. AI vendors classified as business associates requiring BAAs."},
            {"name": "FDA AI/ML Guidance", "detail": "Seven-step risk-based credibility assessment for AI models used in clinical decisions."},
            {"name": "NIST AI RMF", "detail": "Voluntary framework with Govern, Map, Measure, Manage functions across 72 subcategories."},
            {"name": "State Laws (TX, IL, CO)", "detail": "State-specific AI transparency and bias-testing requirements for healthcare decisions."},
        ],
        "required_controls": [
            {"control": "PHI encryption at rest and in transit", "aws_service": "AWS KMS"},
            {"control": "Access audit trails for AI systems", "aws_service": "Amazon CloudWatch + CloudTrail"},
            {"control": "Minimum-necessary data disclosure to AI", "aws_service": "AWS IAM + Bedrock Guardrails"},
            {"control": "Continuous compliance evidence collection", "aws_service": "AWS Audit Manager"},
            {"control": "Security posture monitoring", "aws_service": "AWS Security Hub"},
            {"control": "Bias detection and model explainability", "aws_service": "SageMaker Clarify"},
            {"control": "Human-in-the-loop for clinical decisions", "aws_service": "Amazon A2I"},
        ],
        "penalty": "Up to $2.13M annually per HIPAA violation category",
    },
    "Financial Services": {
        "icon": "🏦",
        "color": "#FF9900",
        "regulations": [
            {"name": "FINRA 2026 Oversight Report", "detail": "GenAI outputs subject to Rule 3110 (supervision), Rule 2210 (communications), and full recordkeeping of prompts, outputs, and model versions."},
            {"name": "SEC AI Guidance", "detail": "AI-washing enforcement — marketing materials must accurately represent AI capabilities."},
            {"name": "US Treasury AI Framework", "detail": "AI Lexicon and Financial Services AI Risk Management Framework for standardized governance."},
            {"name": "SOX / SOC 2", "detail": "Internal controls over financial reporting extend to AI-generated analyses and recommendations."},
        ],
        "required_controls": [
            {"control": "Full audit trail of AI prompts, outputs, and model versions", "aws_service": "Amazon CloudWatch + CloudTrail"},
            {"control": "Encryption of proprietary trading data and PII", "aws_service": "AWS KMS"},
            {"control": "Role-based access to AI models and data", "aws_service": "AWS IAM"},
            {"control": "Continuous compliance assessment", "aws_service": "AWS Audit Manager"},
            {"control": "Centralized security findings", "aws_service": "AWS Security Hub"},
            {"control": "Content filtering and PII redaction", "aws_service": "Bedrock Guardrails"},
            {"control": "Model monitoring for concept drift", "aws_service": "SageMaker Model Monitor"},
        ],
        "penalty": "SEC/FINRA fines + reputational damage + potential license revocation",
    },
    "Life Sciences & Pharma": {
        "icon": "🧬",
        "color": "#16DB93",
        "regulations": [
            {"name": "FDA-EMA Good AI Practice (Jan 2026)", "detail": "10 joint guiding principles requiring traceable inputs, governed outputs, and documented human oversight."},
            {"name": "21 CFR Part 11", "detail": "Electronic records and signatures — applies to GxP-impacting AI in R&D, clinical, and manufacturing."},
            {"name": "EU AI Act (Aug 2026)", "detail": "AI in medical devices classified as high-risk requiring conformity assessments and EU database registration."},
            {"name": "ICH Guidelines", "detail": "International harmonization of AI validation requirements for pharmaceutical development."},
        ],
        "required_controls": [
            {"control": "Traceable data lineage for AI inputs/outputs", "aws_service": "Amazon CloudWatch + CloudTrail"},
            {"control": "Validated electronic records with audit trails", "aws_service": "AWS Audit Manager"},
            {"control": "Encryption of clinical and research data", "aws_service": "AWS KMS"},
            {"control": "Strict access controls for GxP systems", "aws_service": "AWS IAM"},
            {"control": "Continuous security monitoring", "aws_service": "AWS Security Hub"},
            {"control": "Model versioning and reproducibility", "aws_service": "SageMaker Model Registry"},
            {"control": "Bias and fairness assessment", "aws_service": "SageMaker Clarify"},
        ],
        "penalty": "€35M or 7% global turnover (EU AI Act) + FDA warning letters + clinical trial holds",
    },
    "Government & Public Sector": {
        "icon": "🏛️",
        "color": "#232F3E",
        "regulations": [
            {"name": "OMB Memoranda M-25-21/M-25-22", "detail": "Federal AI governance requirements including risk management, transparency, and accountability."},
            {"name": "NIST AI RMF + SP 800-53", "detail": "Combined AI risk management with federal security controls baseline."},
            {"name": "FedRAMP", "detail": "Cloud security authorization required for AI systems processing federal data."},
            {"name": "Executive AI Directives", "detail": "America's AI Action Plan balancing innovation with oversight for federal agencies."},
        ],
        "required_controls": [
            {"control": "FedRAMP-authorized infrastructure", "aws_service": "AWS GovCloud"},
            {"control": "FIPS 140-2 validated encryption", "aws_service": "AWS KMS (GovCloud)"},
            {"control": "Comprehensive access controls and MFA", "aws_service": "AWS IAM"},
            {"control": "Continuous monitoring and logging", "aws_service": "Amazon CloudWatch + CloudTrail"},
            {"control": "Automated compliance assessment", "aws_service": "AWS Audit Manager"},
            {"control": "Centralized security management", "aws_service": "AWS Security Hub"},
            {"control": "Data classification and protection", "aws_service": "Amazon Macie"},
        ],
        "penalty": "Loss of Authority to Operate (ATO) + congressional oversight + public trust erosion",
    },
}

COMPLIANCE_CHECKLIST = {
    "Data Governance": [
        "Training data is encrypted at rest and in transit",
        "Data access follows least-privilege principles",
        "Data lineage is documented and traceable",
        "PII/PHI handling complies with applicable regulations",
        "Data retention and deletion policies are defined",
    ],
    "Model Governance": [
        "Model cards document purpose, limitations, and risk ratings",
        "Model versioning and artifact tracking is in place",
        "Bias and fairness assessments are performed pre-deployment",
        "Human oversight is defined for high-risk decisions",
        "Model performance is continuously monitored for drift",
    ],
    "Access & Identity": [
        "IAM roles follow least-privilege for Bedrock/SageMaker access",
        "MFA is enforced for model deployment and data access",
        "Service-control policies prevent unauthorized model usage",
        "Cross-account access is governed by resource policies",
        "API access is logged and auditable",
    ],
    "Monitoring & Audit": [
        "CloudWatch alarms are set for anomalous AI usage patterns",
        "CloudTrail logs all Bedrock and SageMaker API calls",
        "Guardrail intervention rates are tracked and reviewed",
        "Security Hub findings are reviewed on a regular cadence",
        "Audit Manager assessments are scheduled and delegated",
    ],
    "Incident Response": [
        "Runbooks exist for AI-specific incidents (data poisoning, prompt injection)",
        "Automated remediation is configured for critical security findings",
        "Escalation paths are defined for guardrail bypass attempts",
        "Post-incident review process includes AI-specific root cause analysis",
        "Communication plan covers AI incident disclosure requirements",
    ],
}


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def render_service_card(name, svc):
    """Render an expandable card for a single AWS security service."""
    with st.expander(f"{svc['icon']}  {name} — {svc['tagline']}", expanded=False):
        st.markdown(f"**{svc['description']}**")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("##### 🎯 GenAI Use Cases")
            for uc in svc["genai_use_cases"]:
                st.markdown(f"- {uc}")

            st.markdown("##### ⭐ Key Features")
            for feat in svc["key_features"]:
                st.markdown(f"- {feat}")

        with col_right:
            st.markdown("##### ✅ Best Practices")
            for bp in svc["best_practices"]:
                st.markdown(f"- {bp}")


def build_architecture_graph(selected_services):
    """Build a Plotly network graph showing how selected services protect a GenAI pipeline."""

    # Fixed positions for the core GenAI pipeline (top row)
    pipeline_nodes = [
        ("Training Data\n(S3)", 0, 2),
        ("Model Training\n(SageMaker)", 1.5, 2),
        ("Foundation Model\n(Bedrock)", 3, 2),
        ("Inference\nEndpoint", 4.5, 2),
        ("Application\n/ User", 6, 2),
    ]

    # Security service positions (bottom row)
    all_svc_positions = {
        "AWS IAM":              (0.5, 0),
        "AWS KMS":              (1.8, 0),
        "Amazon CloudWatch":    (3.1, 0),
        "AWS Audit Manager":    (4.4, 0),
        "AWS Security Hub":     (5.7, 0),
    }

    # Connections: which pipeline nodes each service protects
    service_connections = {
        "AWS IAM":           [0, 1, 2, 3, 4],
        "AWS KMS":           [0, 1, 2],
        "Amazon CloudWatch": [1, 2, 3],
        "AWS Audit Manager": [0, 1, 2, 3],
        "AWS Security Hub":  [0, 1, 2, 3, 4],
    }

    fig = go.Figure()

    # Draw pipeline edges
    for i in range(len(pipeline_nodes) - 1):
        x0, y0 = pipeline_nodes[i][1], pipeline_nodes[i][2]
        x1, y1 = pipeline_nodes[i + 1][1], pipeline_nodes[i + 1][2]
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color="#232F3E", width=2),
            hoverinfo="skip", showlegend=False,
        ))

    # Draw security-to-pipeline connections for selected services
    for svc_name in selected_services:
        sx, sy = all_svc_positions[svc_name]
        color = SECURITY_SERVICES[svc_name]["color"]
        for idx in service_connections.get(svc_name, []):
            px_, py_ = pipeline_nodes[idx][1], pipeline_nodes[idx][2]
            fig.add_trace(go.Scatter(
                x=[sx, px_], y=[sy, py_], mode="lines",
                line=dict(color=color, width=1.5, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ))

    # Draw pipeline nodes
    fig.add_trace(go.Scatter(
        x=[n[1] for n in pipeline_nodes],
        y=[n[2] for n in pipeline_nodes],
        mode="markers+text",
        marker=dict(size=40, color="#232F3E", symbol="square"),
        text=[n[0] for n in pipeline_nodes],
        textposition="top center",
        textfont=dict(size=11, color="#232F3E"),
        hoverinfo="text",
        hovertext=[n[0].replace("\n", " ") for n in pipeline_nodes],
        showlegend=False,
    ))

    # Draw selected security service nodes
    if selected_services:
        svc_x = [all_svc_positions[s][0] for s in selected_services]
        svc_y = [all_svc_positions[s][1] for s in selected_services]
        svc_colors = [SECURITY_SERVICES[s]["color"] for s in selected_services]
        svc_labels = [f"{SECURITY_SERVICES[s]['icon']} {s}" for s in selected_services]

        fig.add_trace(go.Scatter(
            x=svc_x, y=svc_y, mode="markers+text",
            marker=dict(size=35, color=svc_colors, symbol="diamond"),
            text=svc_labels,
            textposition="bottom center",
            textfont=dict(size=10),
            hoverinfo="text",
            hovertext=svc_labels,
            showlegend=False,
        ))

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 7]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 3.2]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_industry_detail(industry, data):
    """Render the detail view for a selected industry."""

    st.markdown(f"### {data['icon']}  {industry}")

    # Regulations
    st.markdown("#### 📜 Applicable Regulations")
    for reg in data["regulations"]:
        st.markdown(
            f"""<div style='padding:10px 15px;margin:6px 0;background:#f8f9fa;
            border-left:4px solid {data["color"]};border-radius:4px;'>
            <strong>{reg['name']}</strong><br>
            <span style='color:#555;font-size:0.92em;'>{reg['detail']}</span>
            </div>""",
            unsafe_allow_html=True,
        )

    # Controls → AWS service mapping table
    st.markdown("#### 🔗 Required Controls → AWS Service Mapping")
    controls_df = pd.DataFrame(data["required_controls"])
    controls_df.columns = ["Required Control", "AWS Service"]
    st.dataframe(controls_df, use_container_width=True, hide_index=True)

    # Penalty callout
    st.warning(f"**Non-compliance risk:** {data['penalty']}")


def render_compliance_checklist():
    """Render an interactive compliance readiness checklist with a score."""

    if "checklist_state" not in st.session_state:
        st.session_state.checklist_state = {}

    total = 0
    checked = 0

    for category, items in COMPLIANCE_CHECKLIST.items():
        st.markdown(f"##### {category}")
        for item in items:
            total += 1
            key = f"chk_{category}_{item[:30]}"
            val = st.checkbox(item, key=key, value=st.session_state.checklist_state.get(key, False))
            st.session_state.checklist_state[key] = val
            if val:
                checked += 1

    # Readiness score
    score = int((checked / total) * 100) if total > 0 else 0

    st.markdown("---")
    st.markdown("#### 📈 Compliance Readiness Score")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if score >= 80:
            color = "#3F8624"
            label = "Strong"
        elif score >= 50:
            color = "#FF9900"
            label = "Moderate"
        else:
            color = "#DD344C"
            label = "Needs Attention"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": f"Readiness: {label}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#FFEBEE"},
                    {"range": [50, 80], "color": "#FFF3E0"},
                    {"range": [80, 100], "color": "#E8F5E9"},
                ],
                "threshold": {
                    "line": {"color": "#232F3E", "width": 2},
                    "thickness": 0.8,
                    "value": score,
                },
            },
        ))
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"You've checked **{checked}** of **{total}** controls ({score}%)")


# ──────────────────────────────────────────────
# Tab 1: AWS GenAI Security Services
# ──────────────────────────────────────────────

def tab_security_services():
    st.markdown(custom_header("AWS Generative AI Security Services", 2), unsafe_allow_html=True)

    st.markdown("""
    Securing generative AI workloads requires a layered approach. AWS provides purpose-built 
    services that work together to protect every stage of the GenAI lifecycle — from training 
    data ingestion through model deployment and inference.
    """)

    # ── Service deep-dives ──
    st.markdown("### 🔍 Service Deep Dive")
    st.info("Expand each service below to explore its role in securing GenAI workloads.")

    for name, svc in SECURITY_SERVICES.items():
        render_service_card(name, svc)

    # ── Interactive architecture builder ──
    st.markdown("---")
    st.markdown("### 🏗️ Interactive Architecture Builder")
    st.markdown(
        "Select the security services you want to include and see how they map onto "
        "a typical GenAI pipeline."
    )

    selected = st.multiselect(
        "Choose services to visualize",
        options=list(SECURITY_SERVICES.keys()),
        default=list(SECURITY_SERVICES.keys()),
        key="arch_services",
    )

    fig = build_architecture_graph(selected)
    st.plotly_chart(fig, use_container_width=True)

    if selected:
        st.markdown("#### Coverage Summary")
        coverage_data = []
        pipeline_stages = ["Training Data", "Model Training", "Foundation Model", "Inference Endpoint", "Application"]
        stage_map = {
            "AWS IAM": [True, True, True, True, True],
            "AWS KMS": [True, True, True, False, False],
            "Amazon CloudWatch": [False, True, True, True, False],
            "AWS Audit Manager": [True, True, True, True, False],
            "AWS Security Hub": [True, True, True, True, True],
        }
        for svc in selected:
            row = {"Service": f"{SECURITY_SERVICES[svc]['icon']} {svc}"}
            for i, stage in enumerate(pipeline_stages):
                row[stage] = "✅" if stage_map[svc][i] else "—"
            coverage_data.append(row)

        st.dataframe(pd.DataFrame(coverage_data), use_container_width=True, hide_index=True)
    else:
        st.warning("Select at least one service to see the architecture diagram.")


# ──────────────────────────────────────────────
# Tab 2: Governance & Compliance
# ──────────────────────────────────────────────

def tab_governance_compliance():
    st.markdown(custom_header("Governance & Compliance for Regulated Workloads", 2), unsafe_allow_html=True)

    st.markdown("""
    2026 marks the first year of serious enforcement for AI systems globally. The EU AI Act 
    reaches full enforcement in August 2026, HIPAA has new AI-specific requirements, and 
    financial regulators like FINRA now mandate GenAI governance programs. Organizations 
    deploying generative AI in regulated industries must map their controls to specific 
    regulatory requirements.
    """)

    # ── Regulatory landscape overview ──
    st.markdown("### 🌍 Regulatory Landscape at a Glance")

    landscape_data = {
        "Framework": ["EU AI Act", "HIPAA (2026 update)", "FINRA 2026 Report", "FDA-EMA Good AI Practice", "NIST AI RMF", "ISO 42001"],
        "Scope": ["All high-risk AI in EU", "AI accessing PHI in healthcare", "GenAI in financial services", "AI in drug development", "Cross-industry (voluntary)", "Cross-industry (certifiable)"],
        "Enforcement Date": ["Aug 2, 2026", "Feb 16, 2026", "Active (2026)", "Jan 2026", "Active (voluntary)", "Active (certifiable)"],
        "Max Penalty": ["€35M / 7% revenue", "$2.13M / year", "Fines + license risk", "Clinical holds + warnings", "N/A (voluntary)", "N/A (certification)"],
    }
    st.dataframe(pd.DataFrame(landscape_data), use_container_width=True, hide_index=True)

    # ── Industry framework selector ──
    st.markdown("---")
    st.markdown("### 🏢 Industry-Specific Compliance Requirements")
    st.markdown("Select an industry to see applicable regulations, required controls, and AWS service mappings.")

    industry_cols = st.columns(len(INDUSTRY_FRAMEWORKS))
    selected_industry = st.session_state.get("selected_industry", list(INDUSTRY_FRAMEWORKS.keys())[0])

    for i, (industry, data) in enumerate(INDUSTRY_FRAMEWORKS.items()):
        with industry_cols[i]:
            if st.button(
                f"{data['icon']}  {industry}",
                key=f"ind_{industry}",
                use_container_width=True,
                type="primary" if industry == selected_industry else "secondary",
            ):
                st.session_state.selected_industry = industry
                st.rerun()

    selected_industry = st.session_state.get("selected_industry", list(INDUSTRY_FRAMEWORKS.keys())[0])
    render_industry_detail(selected_industry, INDUSTRY_FRAMEWORKS[selected_industry])

    # ── Compliance readiness checklist ──
    st.markdown("---")
    st.markdown("### ✅ GenAI Compliance Readiness Checklist")
    st.markdown(
        "Walk through this checklist to assess your organization's readiness for deploying "
        "GenAI in a regulated environment. Your score updates in real time."
    )

    render_compliance_checklist()

    # ── Shared responsibility callout ──
    st.markdown("---")
    st.markdown("### 🤝 Shared Responsibility for GenAI")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        #### AWS Responsibilities
        - Physical infrastructure security
        - Service-level encryption capabilities
        - Compliance certifications (SOC, ISO, FedRAMP, HIPAA BAA)
        - Foundation model hosting security (Bedrock)
        - Guardrails and safety tooling
        - Data isolation between tenants
        """)
    with col2:
        st.markdown("""
        #### Customer Responsibilities
        - IAM policies and access management
        - Data classification and encryption key management
        - Model governance (cards, monitoring, bias testing)
        - Regulatory compliance mapping and evidence
        - Prompt engineering and guardrail configuration
        - Human oversight and incident response
        """)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    st.markdown(custom_header("Security, Governance & Compliance for GenAI", 1), unsafe_allow_html=True)

    with st.sidebar:
        common.render_sidebar()

        with st.expander("ℹ️ About This Page", expanded=False):
            st.markdown("""
            This page covers **Domain 5** of the AWS AI Practitioner exam:
            
            - AWS security services for protecting GenAI workloads
            - Governance frameworks for regulated industries
            - Compliance requirements and readiness assessment
            
            Explore the tabs to learn about each topic interactively.
            """)

    tab1, tab2 = st.tabs([
        "🔒 AWS GenAI Security Services",
        "⚖️ Governance & Compliance",
    ])

    with tab1:
        tab_security_services()

    with tab2:
        tab_governance_compliance()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#666;font-size:0.85em;'>"
        "© 2026, Amazon Web Services, Inc. or its affiliates. All rights reserved."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    if "localhost" in st.context.headers["host"]:
        main()
    else:
        is_authenticated = authenticate.login()
        if is_authenticated:
            main()
