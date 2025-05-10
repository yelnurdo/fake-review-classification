import streamlit as st
import torch
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
# from io import StringIO
import base64
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForSequenceClassification



st.set_page_config(
    page_title="Fake Review Detector",
    layout="wide"
)


# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1E88E5;
#         text-align: center;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         color: #0D47A1;
#     }
#     .result-box {
#         padding: 20px;
#         border-radius: 10px;
#         margin-bottom: 20px;
#     }
#     .fake-result {
#         background-color: rgba(255, 99, 71, 0.2);
#         border: 1px solid rgba(255, 99, 71, 0.6);
#     }
#     .real-result {
#         background-color: rgba(46, 204, 113, 0.2);
#         border: 1px solid rgba(46, 204, 113, 0.6);
#     }
# </style>
# """, unsafe_allow_html=True)



@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('alibidos/fake-review-bert')
    model = AutoModelForSequenceClassification.from_pretrained('alibidos/fake-review-bert')
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model.to(device).eval()
    return tokenizer, model, device


tokenizer, model, device = load_model()

sample_real_review = """
This coffee maker exceeded my expectations! After using it daily for three months, I'm impressed by its durability and consistent performance. ...
"""
sample_fake_review = """
OMG!!! BEST PRODUCT EVER!!!!! I just received this amazing coffee maker yesterday and it has already changed my life!!!! ...
"""


if 'text' not in st.session_state:
    st.session_state['text'] = ""


st.markdown("<h1 class='main-header'>Advanced Fake Review Detector</h1>", unsafe_allow_html=True)
tab1, tab2 = st.tabs(["Single Review Analysis", "File Analysis"])


with tab1:
    st.markdown("<h2 class='sub-header'>Analyze a Single Review</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Try Sample Real Review"):
            st.session_state['text'] = sample_real_review
    with col2:
        if st.button("Try Sample Fake Review"):
            st.session_state['text'] = sample_fake_review

    st.session_state['text'] = st.text_area(
        'Enter product review',
        value=st.session_state['text'],
        height=200,
        key='text_area'
    )

    if st.button("Analyze Review"):
        text = st.session_state['text']
        if not text.strip():
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing review..."):
                inputs = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                pred_class = int(probs.argmax())
                label = model.config.id2label[pred_class]

                result_class = "fake-result" if label == "FAKE" else "real-result"
                st.markdown(f"""
                    <div class='result-box {result_class}'>
                        <h3>Prediction Result</h3>
                        <p>This review is classified as: <strong>{label}</strong></p>
                    </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(probs[1] * 100),
                    title={'text': "Probability of being a fake review"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "green"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Probability of Real", f"{probs[0] * 100:.2f}%")
                with col2:
                    st.metric("Probability of Fake", f"{probs[1] * 100:.2f}%")


with tab2:
    st.markdown("<h2 class='sub-header'>Batch Process Multiple Reviews</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a CSV file with reviews", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(data.head())

            col1, col2 = st.columns(2)
            with col1:
                review_col = st.selectbox("Select review column", data.columns)
            with col2:
                label_col_options = ["None"] + list(data.columns)
                actual_label_col = st.selectbox("Select actual label column (if exists)", label_col_options)

            if st.button("Process All Reviews"):
                if len(data) > 100:
                    st.warning(f"You're about to process {len(data)} reviews. This might take some time.")

                progress_bar = st.progress(0)
                with st.spinner(f"Processing {len(data)} reviews..."):
                    results = []
                    batch_size = 16
                    for i in range(0, len(data), batch_size):
                        batch = data[review_col].iloc[i:i + batch_size].tolist()
                        batch_inputs = tokenizer(
                            batch,
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        )
                        batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                        with torch.no_grad():
                            batch_outputs = model(**batch_inputs)
                        batch_probs = torch.softmax(batch_outputs.logits, dim=-1).cpu().numpy()
                        for j, probs in enumerate(batch_probs):
                            pred_class = int(probs.argmax())
                            label = model.config.id2label[pred_class]
                            current_result = {
                                'review': batch[j],
                                'predicted_label': label,
                                'real_prob': probs[0],
                                'fake_prob': probs[1]
                            }

                            if actual_label_col != "None":
                                current_idx = i + j
                                if current_idx < len(data):
                                    current_result['actual_label'] = data[actual_label_col].iloc[current_idx]

                            results.append(current_result)
                        progress_bar.progress(min(1.0, (i + batch_size) / len(data)))

                    results_df = pd.DataFrame(results)

                    st.write("### Results")
                    st.dataframe(results_df)

                    if actual_label_col != "None" and 'actual_label' in results_df.columns:
                        results_df['is_correct'] = results_df['predicted_label'] == results_df['actual_label']
                        accuracy = results_df['is_correct'].mean() * 100

                        unique_labels = set(results_df['predicted_label'].unique()) | set(
                            results_df['actual_label'].unique())
                        confusion_counts = {}
                        for actual in unique_labels:
                            confusion_counts[actual] = {}
                            for predicted in unique_labels:
                                subset = results_df[(results_df['actual_label'] == actual) &
                                                    (results_df['predicted_label'] == predicted)]
                                confusion_counts[actual][predicted] = len(subset)

                        st.write(f"### Model Accuracy: {accuracy:.2f}%")

                        st.write("### Confusion Matrix")
                        confusion_df = pd.DataFrame(confusion_counts).fillna(0).astype(int)
                        st.dataframe(confusion_df)

                        st.write("### Class-wise Performance")
                        metrics_data = []
                        for label in unique_labels:
                            tp = confusion_counts.get(label, {}).get(label, 0)
                            actual_total = results_df['actual_label'].eq(label).sum()
                            predicted_total = results_df['predicted_label'].eq(label).sum()

                            precision = tp / predicted_total if predicted_total > 0 else 0
                            recall = tp / actual_total if actual_total > 0 else 0
                            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                            metrics_data.append({
                                "Label": label,
                                "Precision": f"{precision:.2f}",
                                "Recall": f"{recall:.2f}",
                                "F1 Score": f"{f1:.2f}"
                            })

                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df)


                    unique_predicted_labels = set(result['predicted_label'] for result in results)
                    label_counts = {}
                    for label in unique_predicted_labels:
                        label_counts[label] = sum(1 for result in results if result['predicted_label'] == label)

                    if 'FAKE' in unique_predicted_labels and 'REAL' in unique_predicted_labels:
                        fake_count = label_counts.get('FAKE', 0)
                        real_count = label_counts.get('REAL', 0)
                    elif 'CG' in unique_predicted_labels and 'OR' in unique_predicted_labels:

                        fake_count = label_counts.get('CG', 0)
                        real_count = label_counts.get('OR', 0)

                    st.write("### Distribution of Predictions")

                    st.write("### Review Statistics")
                    if 'FAKE' in unique_predicted_labels and 'REAL' in unique_predicted_labels:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Fake Reviews", fake_count)
                        with col2:
                            st.metric("Real Reviews", real_count)
                    elif 'CG' in unique_predicted_labels and 'OR' in unique_predicted_labels:
                        st.write("Found labels: 'CG' and 'OR'")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("CG Reviews (Likely Fake)", fake_count)
                        with col2:
                            st.metric("OR Reviews (Likely Real)", real_count)
                    else:
                        st.write(f"Found labels: {', '.join(unique_predicted_labels)}")
                        for label, count in label_counts.items():
                            st.metric(f"{label} Reviews", count)




        except Exception as e:
            st.error(f"Error processing file: {e}")