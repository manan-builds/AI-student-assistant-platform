import gradio as gr
from transformers import pipeline
from utils.pdfReader import extract_text
from utils.preprocess import clean_text, chunk_text

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment = pipeline("sentiment-analysis")

def process_file(file, question):
    text = extract_text(file.name)
    text = clean_text(text)

    chunks = chunk_text(text)
    summary = summarizer(text[:1000])[0]["summary_text"]

    best_answer = ""
    best_score = 0

    for ch in chunks[:5]:
        result = qa_pipeline(question=question, context=ch)
        if result["score"] > best_score:
            best_score = result["score"]
            best_answer = result["answer"]

    senti = sentiment(text[:500])[0]
    return summary, best_answer, senti
    
ui = gr.Interface(
    fn=process_file,
    inputs=[
        gr.File(label="Upload PDF"),
        gr.Textbox(label="Ask a Question")
    ],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Answer"),
        gr.JSON(label="Sentiment")
    ],
    title="AI Student Assistant Platform"
)

ui.launch()

