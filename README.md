# MDSK-RAG Demonstration Files

This repository contains demonstration resources for the manuscript:

**"Materials Dual-Source Knowledge Retrieval-Augmented Generation for Local Large Language Models in Photocatalysts"**  
*Journal of Chemical Information and Modeling (JCIM)*.  https://doi.org/10.1021/acs.jcim.5c01941

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)


## Contents

### 1. `mdsk-rag-example.ipynb`
A Jupyter Notebook demonstrating the core MDSK-RAG framework.

### 2. `mdsk-rag-chatbot-app-example.py`
A Streamlit-based web application providing an interactive interface.

### 3. `requirements.txt`
List of Python dependencies and their versions from the execution environment.  

### 4. `selected-experimental-data.csv`
An example of the structured experimental data used in the study.  
**Note:** When processed through the MDSK-RAG framework using the predefined flowchart and sentence templates, this data produces the results shown in **Box 2** of the manuscript.

## How to Use

1. **Prepare required files and variables**  
   Before running the demonstration files, please prepare the following:
   - **`HF_TOKEN`**: Your Hugging Face authentication token. 
   - **`YOUR_CSV_FILE_PATH`**: Path to the CSV file containing the structured experimental data.  
   - **`YOUR_PDF_FILE_PATHS`**: Path or glob pattern to the PDF files containing the scientific literature.
   - **`YOUR_SUMMARY_PROMPT`**: A text prompt used for summarization in the MDSK-RAG.  
  The prompt is designed to guide summaries toward domain-relevant information.  
  In the present study, the focus is on composition, crystal structure type, synthesis method, reaction conditions, and hydrogen evolution activity.  
  The actual prompt used is: 
  *"Please summarize the following text, focusing on the composition, crystal structure type, synthesis method, reaction conditions, and hydrogen evolution activity:"*
   - **`YOUR_QUESTION`**: The question you want to input.  

2. **Set up a virtual environment for this demonstration and install the required libraries**  
   Using pip:
   ```bash
   pip install -r requirements.txt
   ``` 
   Using uv:
      ```bash
   uv pip install -r requirements.txt
   ``` 

3. Run the Jupyter Notebook for the step-by-step demonstration:  
   Open `mdsk-rag-example.ipynb` with your preferred editor.

4. Launch the web application:
   ```bash
   streamlit run mdsk-rag-chatbot-app-example.py
   ```
## Execution Environment

The demonstration files were tested with **Python 3.10.12** on **Ubuntu 22.04.2 LTS** running under **WSL2** on Windows 11.
