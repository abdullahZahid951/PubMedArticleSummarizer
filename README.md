# PubMed Article Summarizer

This project converts PubMed medical articles into summaries using the T5 model. It allows users to input PubMed articles and get concise abstracts as output.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- Converts PubMed articles into concise summaries.
- Utilizes the T5 model for natural language processing.
- Easy to use with a simple command-line interface.

## Installation

### Prerequisites

- Anaconda distribution with Python 3.8 or higher installed.

### Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/pubmed-article-summarizer.git
    cd pubmed-article-summarizer
    ```

2. **Create a new Anaconda environment and activate it:**

    ```sh
    conda create --name pubmed_summarizer python=3.8
    conda activate pubmed_summarizer
    ```

3. **Install the required dependencies:**

    ```sh
    conda install -c huggingface datasets  # For handling datasets
    conda install -c conda-forge spacy     # For NLP tasks
    python -m spacy download en_core_web_sm # Spacy English core model
    pip install torch torchvision torchaudio  # PyTorch for deep learning
    conda install -c huggingface transformers  # Hugging Face Transformers library
    conda install -c conda-forge sentencepiece  # SentencePiece for tokenization
    pip install streamlit
    ```

### Dependencies

- **`datasets`**: Provides an interface to handle datasets, which is useful for managing PubMed articles.
  
- **`spacy`**: A library for natural language processing tasks such as tokenization and language model loading. The `en_core_web_sm` model is downloaded for tokenization.

- **`torch`, `torchvision`, `torchaudio`**: Libraries from PyTorch, a popular deep learning framework, used for building and training neural networks.

- **`transformers`**: A library from Hugging Face providing pre-trained models and tools for natural language understanding tasks, including the T5 model.

- **`sentencepiece`**: A library used for tokenization, particularly suited for models like T5 that employ subword tokenization.

These dependencies are crucial for the project's functionality, enabling tasks such as data handling, text processing, model training, and inference.

## Usage

### Option 1: Using a Pre-trained Model

1. **Download the pre-trained model (`fine_tuned_t5_model`):**

    - Download the pre-trained model folder and place it in the main directory of your device.
    - Then run app.py file with `streamlit run app.py` 

2. **Activate the Anaconda environment (if not already activated):**

    ```sh
    conda activate pubmed_summarizer
    ```

3. **Run the summarizer script:**

    ```sh
    python summarize.py --input path/to/pubmed_article.txt --output path/to/summary.txt --model_dir fine_tuned_t5_model
    ```

    - `--input` : Path to the input file containing the PubMed article.
    - `--output` : Path to the output file where the summary will be saved.
    - `--model_dir` : Path to the directory containing the pre-trained model.

### Option 2: Running and Tuning the Model via Jupyter Notebook

1. **Activate the Anaconda environment (if not already activated):**

    ```sh
    conda activate pubmed_summarizer
    ```

2. **Run Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

3. **Open and run the provided notebook (`summarize.ipynb`):**

    - Navigate to the `pubmed-article-summarizer` directory in Jupyter Notebook.
    - Open `summarize.ipynb`.
    - Follow the instructions in the notebook to configure settings, tune the model, and summarize a PubMed article.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for the T5 model and the `transformers` library.
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/) for providing access to medical articles.
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) for web scraping utilities.
