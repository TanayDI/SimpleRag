# SimpleRag
This repository contains simple RAG projects that might help you understand the concepts of RAG in a simple way.
This Project includes 3 python files, each implementing RAG principles and workflow.
#### PDFRag:
It takes a pdf as an input and answers questions based on the pdf data.
#### GardenRamsey
Ever had a thought of what to cook tonight? Don't worry just give GardenRamsey a list of ingredients you have and it will give you a recipe based on your preferred cuisine. It searches a variety of recipies on the web based on the ingredients. If you have a cookbook you can use a cookbook as well in the from of pdf for reference.
#### YoutubeRag
As the chatbot quetions related to a youtube video you are watching. It answers the questions based on the transcription of the video. Just paste the youtube link.

---

### Usage
1. Clone the repository
  ```bash
  git clone https://github.com/TanayDI/SimpleRag.git
  ```
2. Make a python environment.
  ```bash
  python -m venv .venv
  .\.venv\Scripts\activate
  ```
3. Install dependencies
  ```bash
  pip install requirements.txt
  ```
4. Update the existing or make a new ``.env`` file with your gen-ai api keys. You can get the keys from https://aistudio.google.com/apikey
  ```.env
  GOOGLE_API_KEY=YOUR_API_KEY
  ```
5. Run the files using streamlit.
```bash
streamlit run PDFRag.py
streamlit run GardenRamsey.py
streamlit run YoutubeRag.py
```

---

#### Give this project a star if you like it :D
