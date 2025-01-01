# AnswerGenie

AnswerGenie is an AI-powered application designed to answer questions based on the content of uploaded documents. It uses semantic search to find relevant information in the documents and leverages a large language model (LLM) to generate accurate and helpful responses.

## Features

- Upload and process various document formats including PDF, DOCX, TXT, PPTX, and images.
- Perform semantic search to find relevant information in the uploaded documents.
- Generate answers to user queries using a large language model.
- Supports both local document search and global search using the LLM.

## Setup

### Prerequisites

- Python 3.8 or higher
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for image processing)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/AnswerGenie.git
    cd AnswerGenie
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root directory and add your Groq API key:
    ```plaintext
    GROQ_API_KEY="your_api_key_here"
    ```

## Usage

### Running the Streamlit App

1. Start the Streamlit app:
    ```sh
    streamlit run RagStreamlit.py
    ```

2. Open your web browser and go to `http://localhost:8501`.

3. Upload your documents using the file uploader.

4. Enter your questions in the input box and get answers based on the uploaded documents.


## File Descriptions

- `RagStreamlit.py`: Streamlit app for uploading documents and asking questions.
- `complete.py`: GUI application for uploading documents and asking questions.
- `chatbot.py`: Terminal-based chatbot for asking questions.
- `vectorDB.py`: Script for managing the vector database and performing semantic search.
- `semanticSearchGUI.py`: GUI for semantic search.
- `getText.py`: Script for extracting text from various document formats.
- `requirements.txt`: List of required Python packages.
- `.gitignore`: Git ignore file to exclude unnecessary files from version control.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.