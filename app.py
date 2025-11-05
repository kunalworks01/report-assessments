import streamlit as st
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

st.set_page_config(page_title="🧠 Mental Health Assessment", layout="centered")
st.title("🧠 Modular Mental Health Assessment")

# Scoring maps for each scale
scoring_maps = {
    "PHQ-9": {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3},
    "GAD-7": {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3},
    "PCL-5": {"Not at all": 0, "A little bit": 1, "Moderately": 2, "Quite a bit": 3, "Extremely": 4},
    "MDQ": {"Yes": 1, "No": 0},
    "ASRS v1.1": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Very often": 4}
}

# Initialize summary state
if "summary" not in st.session_state:
    st.session_state.summary = {}

# Load available JSON scales
scales = [f for f in os.listdir() if f.endswith(".json")]
selected_scale = st.selectbox("📋 Choose a scale to begin:", scales)

# Load selected scale
with open(selected_scale, "r") as f:
    data = json.load(f)

responses = []
st.subheader(f"🧪 Assessment: {data['scale']}")
for i, item in enumerate(data["questions"], 1):
    response = st.radio(f"{i}. {item['text']}", item["options"], key=f"{selected_scale}_{i}")
    responses.append(response)

# Submit and score
if st.button("✅ Submit Responses"):
    total_score = 0
    scale_name = data["scale"]
    score_map = scoring_maps.get(scale_name, {})

    for response in responses:
        total_score += score_map.get(response, 0)

    st.success("Responses submitted!")
    st.write(f"🧮 Total Score for {scale_name}: **{total_score}**")

    # Severity interpretation
    if scale_name == "PHQ-9":
        if total_score <= 4:
            severity = "Minimal depression"
        elif total_score <= 9:
            severity = "Mild depression"
        elif total_score <= 14:
            severity = "Moderate depression"
        elif total_score <= 19:
            severity = "Moderately severe depression"
        else:
            severity = "Severe depression"
        st.info(f"📈 Severity Level: **{severity}**")

    elif scale_name == "GAD-7":
        if total_score <= 4:
            severity = "Minimal anxiety"
        elif total_score <= 9:
            severity = "Mild anxiety"
        elif total_score <= 14:
            severity = "Moderate anxiety"
        else:
            severity = "Severe anxiety"
        st.info(f"📈 Severity Level: **{severity}**")

    # Store in summary
    st.session_state.summary[scale_name] = total_score

# Show summary dashboard
if st.button("📊 Show Summary Dashboard"):
    st.subheader("🧠 Assessment Summary")
    summary_df = pd.DataFrame.from_dict(st.session_state.summary, orient='index', columns=['Score'])
    st.dataframe(summary_df)

    # Bar chart
    st.subheader("📊 Score Distribution")
    st.bar_chart(summary_df)

# Divider
st.divider()

# NLP Section
st.subheader("🧠 Symptom Classifier (NLP)")
user_input = st.text_area("Describe your symptoms or feelings in your own words:")

# Load general-purpose classifier
@st.cache_resource
def load_general_classifier():
    return pipeline("text-classification", model="distilbert-base-uncased", return_all_scores=True)

# Load MentalBERT classifier
@st.cache_resource
def load_mentalbert_classifier():
    tokenizer = AutoTokenizer.from_pretrained("Sharath45/MENTALBERT_MULTILABEL_CLASSIFICATION")
    model = AutoModelForSequenceClassification.from_pretrained("Sharath45/MENTALBERT_MULTILABEL_CLASSIFICATION")
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

# Choose model
model_choice = st.radio("Choose NLP model:", ["General (DistilBERT)", "MentalBERT"], horizontal=True)

if st.button("🔍 Analyze Text"):
    with st.spinner("Analyzing..."):
        if model_choice == "General (DistilBERT)":
            classifier = load_general_classifier()
            label_map = {}  # DistilBERT uses POSITIVE/NEGATIVE directly
        else:
            classifier = load_mentalbert_classifier()
            label_map = {
                "LABEL_0": "Neutral / No distress",
                "LABEL_1": "Possible emotional distress"
            }

        result = classifier(user_input)
        st.write("🧠 Classification Results:")
        labels = []
        scores = []

        for label in result[0]:
            readable = label_map.get(label["label"], label["label"])
            labels.append(readable)
            scores.append(round(label["score"] * 100, 2))
            st.write(f"🔹 {readable}: {round(label['score'] * 100, 2)}%")

        # Pie chart
        st.subheader("📈 NLP Classification Breakdown")
        fig, ax = plt.subplots()
        ax.pie(scores, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
