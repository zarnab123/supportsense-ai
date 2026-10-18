import streamlit as st

from app import get_model
from src.predict import confidence_label, suggest_priority
from src.retriever import retrieve_guidance


st.set_page_config(
    page_title="SupportSense AI",
    page_icon="🤝",
    layout="wide",
)

st.title("🤝 SupportSense AI")
st.subheader("Banking Customer-Support Intelligence Copilot")
st.caption(
    "Classifies support tickets, assigns priority, and retrieves relevant support guidance."
)

with st.sidebar:
    st.header("Project capabilities")
    st.markdown(
        """
        - ML ticket classification
        - Priority prediction
        - Confidence indicator
        - RAG policy guidance
        - Human-review safeguard
        """
    )

    st.divider()
    st.caption("Built with Python, scikit-learn, FastAPI, and Streamlit.")

ticket_text = st.text_area(
    "Customer support ticket",
    value="Someone used my card without permission and I need urgent help.",
    height=140,
    placeholder="Type a customer issue here...",
)

if st.button("Analyze ticket", type="primary", use_container_width=True):
    clean_text = ticket_text.strip()

    if len(clean_text) < 5:
        st.error("Please enter at least 5 characters.")
    else:
        with st.spinner("Analyzing ticket..."):
            model = get_model()
            category = model.predict([clean_text])[0]
            priority = suggest_priority(clean_text, category)
            confidence = confidence_label(model, clean_text)
            guidance = retrieve_guidance(clean_text)

        category_display = category.replace("_", " ").title()

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted category", category_display)
        col2.metric("Priority", priority)
        col3.metric("Confidence", confidence)

        if priority == "HIGH":
            st.error("High-priority case: human support review is required.")
        elif priority == "MEDIUM":
            st.warning("Medium-priority case: queue for timely human review.")
        else:
            st.success("Low-priority case: standard support workflow can be used.")

        st.subheader("Recommended support guidance")
        st.write(f"**Policy:** {guidance['policy_title']}")
        st.markdown(guidance["guidance"])

        st.caption(
            f"Retrieval similarity: {guidance['similarity_score']} | "
            "AI output is a recommendation, not an automatic final decision."
        )

st.divider()
st.caption("Demo project — SupportSense AI")