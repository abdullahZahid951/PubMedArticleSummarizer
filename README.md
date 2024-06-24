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

- Anaconda distribution with Python 3.8 or higher

### Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/pubmed-article-summarizer.git
    cd pubmed-article-summarizer
    ```

2. **Create a new Anaconda environment:**

    ```sh
    conda create --name pubmed_summarizer python=3.8
    conda activate pubmed_summarizer
    ```

3. **Install the required dependencies:**

    ```sh
    conda install -c huggingface datasets
    conda install -c conda-forge spacy
    python -m spacy download en_core_web_sm
    pip install torch torchvision torchaudio
    conda install -c huggingface transformers
    conda install -c conda-forge sentencepiece
    ```

### Dependencies

- `transformers` - Hugging Face library for natural language processing.
- `torch` - PyTorch library for deep learning.
- `requests` - Library for making HTTP requests (for fetching PubMed articles).
- `beautifulsoup4` - Library for parsing HTML and XML documents (for scraping PubMed articles).

## Usage

You have two options to use the summarizer:

### Option 1: Using a Pre-trained Model

1. **Download the pre-trained model:**

    - Download the pre-trained model folder (`fine_tuned_t5_model`) and place it in the main directory of your device (for example for windows C Drive).
    - And after doing you simply run the python script called

2. **Activate the Anaconda environment (if not already activated):**

    ```sh
    conda activate pubmed_summarizer
    ```

3. **Run the summarizer script:**

    ```sh
    python summarize.py --input path/to/pubmed_article.txt --output path/to/summary.txt --model_dir t5_pretrained
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

3. **Open and run the provided notebook:**

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
