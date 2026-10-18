import streamlit as st

from app import get_model
from src.predict import confidence_label, suggest_priority
from src.retriever import retrieve_guidance


st.set_page_config(
    page_title="SupportSense AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            --navy: #102A43;
            --blue: #2563EB;
            --ink: #1F2937;
            --muted: #667085;
            --line: #E4E7EC;
            --canvas: #F6F8FC;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 248, 252, 0.88);
        }

        [data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.5rem;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 0.35rem;
        }

        .brand-icon {
            width: 38px;
            height: 38px;
            border-radius: 10px;
            background: var(--navy);
            color: #FFFFFF;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.82rem;
            font-weight: 800;
        }

        .brand-title {
            color: var(--navy);
            font-size: 1.18rem;
            font-weight: 800;
        }

        .sidebar-text {
            color: var(--muted);
            font-size: 0.85rem;
            line-height: 1.55;
        }

        .sidebar-label {
            color: #667085;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.12rem;
            margin: 1.1rem 0 0.55rem;
        }

        .hero {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 2rem 2.2rem;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.05);
            margin-bottom: 1.45rem;
        }

        .eyebrow {
            color: var(--blue);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.13rem;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            color: var(--navy);
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.05rem;
            line-height: 1.1;
            margin-bottom: 0.65rem;
        }

        .hero-description {
            color: #5E6B7E;
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 760px;
        }

        .stats {
            display: flex;
            gap: 0.7rem;
            flex-wrap: wrap;
            margin-top: 1.35rem;
        }

        .stat {
            background: #F8FAFC;
            border: 1px solid #E4EAF2;
            border-radius: 9px;
            color: #506078;
            padding: 0.52rem 0.75rem;
            font-size: 0.82rem;
        }

        .stat strong {
            color: var(--navy);
        }

        .section-label {
            color: #667085;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.12rem;
            margin: 1.45rem 0 0.55rem;
        }

        .field-title {
            color: var(--navy);
            font-size: 1rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .field-help {
            color: var(--muted);
            font-size: 0.87rem;
            margin-bottom: 0.7rem;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border: 1px solid var(--line) !important;
            border-radius: 16px !important;
            box-shadow: 0 6px 18px rgba(16, 42, 67, 0.035);
        }

        .stTextArea textarea {
            background: #FFFFFF !important;
            color: #1F2937 !important;
            border: 1px solid #C9D3E0 !important;
            border-radius: 10px !important;
            font-size: 0.96rem !important;
        }

        .stTextArea textarea:focus {
            border-color: var(--blue) !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        }

        button[kind="primary"],
        [data-testid="stBaseButton-primary"] {
            background: var(--navy) !important;
            border: 1px solid var(--navy) !important;
            border-radius: 9px !important;
            min-height: 2.85rem !important;
            font-weight: 700 !important;
        }

        button[kind="primary"] *,
        [data-testid="stBaseButton-primary"] * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        button[kind="primary"]:hover,
        [data-testid="stBaseButton-primary"]:hover {
            background: #1D4E7A !important;
            border-color: #1D4E7A !important;
        }

        button[kind="secondary"],
        [data-testid="stBaseButton-secondary"] {
            background: #FFFFFF !important;
            border: 1px solid #D9E1EC !important;
            border-radius: 8px !important;
            min-height: 2.65rem !important;
            font-weight: 600 !important;
        }

        button[kind="secondary"] *,
        [data-testid="stBaseButton-secondary"] * {
            color: #334E68 !important;
        }

        button[kind="secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover {
            background: #F4F7FB !important;
            border-color: #A9BDD5 !important;
        }

        .result-card {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-radius: 13px;
            padding: 1.15rem 1.2rem;
            min-height: 110px;
        }

        .result-label {
            color: #758197;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.1rem;
        }

        .result-value {
            color: var(--navy);
            font-size: 1.25rem;
            font-weight: 800;
            line-height: 1.25;
            margin-top: 0.55rem;
        }

        .guidance-card {
            background: #FFFFFF;
            border: 1px solid var(--line);
            border-left: 4px solid var(--blue);
            border-radius: 12px;
            padding: 1.25rem;
            margin-top: 1rem;
        }

        .guidance-title {
            color: var(--navy);
            font-size: 1.1rem;
            font-weight: 800;
            margin-top: 0.4rem;
        }

        .footer {
            color: #8793A6;
            font-size: 0.82rem;
            text-align: center;
            padding-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "ticket_text" not in st.session_state:
    st.session_state.ticket_text = (
        "Someone used my card without permission and I need urgent help."
    )


def load_demo_ticket(ticket):
    st.session_state.ticket_text = ticket


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">SS</div>
            <div class="brand-title">SupportSense AI</div>
        </div>
        <div class="sidebar-text">Customer-support intelligence workspace</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown('<div class="sidebar-label">DEMO TICKETS</div>', unsafe_allow_html=True)

    st.button(
        "Card used without permission",
        use_container_width=True,
        on_click=load_demo_ticket,
        args=("Someone used my card without permission and I need urgent help.",),
    )
    st.button(
        "ATM retained my card",
        use_container_width=True,
        on_click=load_demo_ticket,
        args=("The ATM kept my card and I cannot get it back.",),
    )
    st.button(
        "Bank transfer is pending",
        use_container_width=True,
        on_click=load_demo_ticket,
        args=("My bank transfer has been pending since yesterday.",),
    )

    st.divider()
    st.markdown('<div class="sidebar-label">CAPABILITIES</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-text">
            Ticket classification<br>
            Priority assessment<br>
            Confidence indicator<br>
            Policy retrieval (RAG)<br>
            Human-review safeguard
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("Demo only — do not enter real banking credentials.")


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-ASSISTED TICKET TRIAGE</div>
        <div class="hero-title">SupportSense AI</div>
        <div class="hero-description">
            Convert unstructured banking support messages into a predicted category,
            priority level, and relevant support guidance.
        </div>
        <div class="stats">
            <div class="stat"><strong>84%</strong> test accuracy</div>
            <div class="stat"><strong>77</strong> banking intent categories</div>
            <div class="stat"><strong>Human review</strong> for every recommendation</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">TICKET WORKSPACE</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.markdown('<div class="field-title">Describe the customer issue</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="field-help">Write the customer message as received by the support team.</div>',
        unsafe_allow_html=True,
    )

    ticket_text = st.text_area(
        "Ticket text",
        key="ticket_text",
        height=135,
        label_visibility="collapsed",
        placeholder="Example: My card was used without my permission.",
    )

    analyze = st.button(
        "Analyze ticket",
        type="primary",
        use_container_width=True,
    )

if analyze:
    clean_text = ticket_text.strip()

    if len(clean_text) < 5:
        st.error("Please enter at least 5 characters.")
    else:
        with st.spinner("Analyzing ticket and retrieving support guidance..."):
            model = get_model()
            category = model.predict([clean_text])[0]
            priority = suggest_priority(clean_text, category)
            confidence = confidence_label(model, clean_text)
            guidance = retrieve_guidance(clean_text)

        category_display = category.replace("_", " ").title()

        st.markdown('<div class="section-label">TRIAGE RESULT</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">PREDICTED CATEGORY</div>
                    <div class="result-value">{category_display}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">PRIORITY LEVEL</div>
                    <div class="result-value">{priority}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">MODEL CONFIDENCE</div>
                    <div class="result-value">{confidence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if priority == "HIGH":
            st.error("High-priority case: human support review is required.")
        elif priority == "MEDIUM":
            st.warning("Medium-priority case: queue for timely human review.")
        else:
            st.success("Low-priority case: use the standard support workflow.")

        st.markdown(
            f"""
            <div class="guidance-card">
                <div class="result-label">RETRIEVED SUPPORT GUIDANCE</div>
                <div class="guidance-title">{guidance["policy_title"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(guidance["guidance"])
        st.caption(
            f"Retrieval similarity: {guidance['similarity_score']} · "
            "Every AI recommendation must be reviewed by a human support agent."
        )

st.markdown(
    '<div class="footer">SupportSense AI · Python · scikit-learn · FastAPI · RAG · Streamlit</div>',
    unsafe_allow_html=True,
)