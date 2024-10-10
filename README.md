# Multimodal Disaster Response

This project implements a multi-model disaster response prediction system using pre-trained SoTA Embeddings models ([MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)) from Hugging Face. It provides a Gradio interface for easy interaction and prediction.

## Installation

1. Clone the repository and navigate to the project folder:
   ```
   git clone https://github.com/s-smits/multimodal-disaster-response
   cd multimodal-disaster-response
   ```

2. Create a virtual environment named `venv_disaster_response` and activate it:
   - For macOS and Linux:
     ```
     python3 -m venv venv_disaster_response
     source venv_disaster_response/bin/activate
     ```
   - For Windows:
     ```
     python -m venv venv_disaster_response
     venv_disaster_response\Scripts\activate
     ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your Hugging Face token as an environment variable:
   ```
   export HF_TOKEN=your_huggingface_token_here
   ```

   Make sure to enter your own Hugging Face token here. This token is necessary for downloading the pre-trained models used in this project.

## Usage

Start the script:
```
python main.py
```

This opens a Gradio interface where you can:
1. Enter up to 10 disaster-related texts
2. Get predictions for each text, including:
   - Timeframe (Preparedness, Response, Other)
   - Transfer type (Request, Provide, Other)
   - Disaster response labels (with multiple thresholds)
   - Overall relevance

## Models

The project uses three pre-trained models:
- Timeframe model: `ssmits/best-timeframe-model-disaster-response`
- Transfer type model: `ssmits/best-transfer-type-model-disaster-response`
- Disaster response model: `ssmits/best-actionable-labelling-model-disaster-response`

## Requirements

See `requirements.txt` for a full list of dependencies. Key libraries include:
- gradio
- numpy
- transformers
- tensorflow
- huggingface_hub

## License

[MIT License](LICENSE)
