import gradio as gr
import numpy as np
import warnings
from transformers import AutoTokenizer, AutoConfig, TFAutoModel, TFBertModel
import os
from huggingface_hub import HfApi, HfFolder, hf_hub_download
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope

# Retrieve the token from the environment variable
hf_token = os.getenv("HF_TOKEN")

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Define pre-trained model names
timeframe_model_name = 'ssmits/best-timeframe-model-disaster-response'
transfer_model_name = 'ssmits/best-transfer-type-model-disaster-response'
disaster_response_model_name = 'ssmits/best-actionable-labelling-model-disaster-response'

# Load tokenizers
timeframe_tokenizer = AutoTokenizer.from_pretrained(timeframe_model_name)
transfer_tokenizer = AutoTokenizer.from_pretrained(transfer_model_name)
disaster_response_tokenizer = AutoTokenizer.from_pretrained(disaster_response_model_name)

# Load configurations
timeframe_config = AutoConfig.from_pretrained(timeframe_model_name)
transfer_config = AutoConfig.from_pretrained(transfer_model_name)
disaster_response_config = AutoConfig.from_pretrained(disaster_response_model_name)

labels_timeframe = ["Preparedness", "Response", "Other"]
labels_transfer = ["Request", "Provide", "Other"]
labels_disaster_response = ["Request-GoodsServices", "Request-SearchAndRescue", "Request-InformationWanted", "CallToAction-Volunteer", "CallToAction-Donations", "CallToAction-MovePeople", "Report-PartyObservation", "Report-EmergingThreats", "Report-ServiceAvailable"]
threshold_disaster_response = 0.12

default_texts = [
    "Urgent help needed in Johnson City! Our neighborhood is completely flooded, many people trapped. Please send rescue teams ASAP! #FloodRelief",
    "@RedCross Our family is stranded on the roof of our home in Springfield. 4 adults, 2 children. We need immediate evacuation assistance. Please help!",
    "If anyone can provide bottled water, dry foods, blankets etc. please drop off supplies at 123 Main St. Dayton. Shelter is housing over 200 flood victims. #FloodRelief",
    "Lost everything in the flooding. House is destroyed. Anyone who can temporarily host a family of 5? We have nowhere to go. #FloodVictims",
    "@FEMA The bridge on Hwy 10 has been completely washed out by floodwaters cutting off our town. Need emergency access and supplies immediately.",
    "Volunteers needed to help sandbag riverbanks and homes in Riverside County. Bring shovels if you can! Meet at the fire station at 10am. #FloodPrep",
    "Our senior center is flooded with 3ft of water. We're trapped on the 2nd floor with 40 residents, most in wheelchairs. In desperate need of rescue boats!",
    "RT and spread: St Mark's Church (564 Broad St) is open as an emergency flood shelter. Need donations of food, water, blankets, baby supplies etc.",
]

def tokenize_text(text, tokenizer, max_length):
    return tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf',
        return_attention_mask=True
    )

def get_predictions_ensemble(x_test, model, batch_size=4):
    predicted_labels_list = []
    for i in range(0, len(x_test), batch_size):
        end = min(i+batch_size, len(x_test))
        batches = x_test[i:end]
        for batch in batches:
            batch_dict = {}
            for key in batch.keys():
                batch_dict[key] = batch[key].numpy()
            y_pred = model.predict(batch_dict)
            predicted_label = np.argmax(y_pred, axis=1)
            predicted_labels_list.extend(predicted_label)
    return predicted_labels_list

def get_predictions_one_hot(x_test, model, batch_size=4):
    predicted_labels_list = []
    for i in range(0, len(x_test), batch_size):
        end = min(i+batch_size, len(x_test))
        batches = x_test[i:end]
        for batch in batches:
            batch_dict = {}
            for key in batch.keys():
                batch_dict[key] = batch[key].numpy()
            y_pred = model.predict(batch_dict)
            predicted_labels_list.extend(y_pred)
    return predicted_labels_list

def load_model_with_custom_objects(model_name, model_file):
    model_path = hf_hub_download(repo_id=model_name, filename=model_file, token=hf_token)
    with custom_object_scope({'TFBertModel': TFBertModel}):
        model = load_model(model_path)
    return model

# Load models globally
timeframe_model_file = "tf_model.h5"
best_timeframe_model = load_model_with_custom_objects(timeframe_model_name, timeframe_model_file)
transfer_model_file = "tf_model.h5"
best_transfer_model = load_model_with_custom_objects(transfer_model_name, transfer_model_file)
disaster_response_model_file = "tf_model.h5"
best_disaster_response_model = load_model_with_custom_objects(disaster_response_model_name, disaster_response_model_file)

def predict(*texts):
    results = []
    thresholds = [0.2, 0.4, 0.6]  # Define multiple thresholds

    for text in texts:
        if text.strip():  # Only process non-empty inputs
            # Tokenize and predict for each model
            tokenized_timeframe = tokenize_text(text, timeframe_tokenizer, max_length=318)
            predicted_labels_timeframe = best_timeframe_model.predict({'input_ids': tokenized_timeframe['input_ids'], 'attention_mask': tokenized_timeframe['attention_mask']})
            timeframe_prediction = labels_timeframe[np.argmax(predicted_labels_timeframe)]
            
            tokenized_transfer = tokenize_text(text, transfer_tokenizer, max_length=171)
            predicted_labels_transfer = best_transfer_model.predict({'input_ids': tokenized_transfer['input_ids'], 'attention_mask': tokenized_transfer['attention_mask']})
            transfer_prediction = labels_transfer[np.argmax(predicted_labels_transfer)]
            
            tokenized_disaster_response = tokenize_text(text, disaster_response_tokenizer, max_length=107)
            predicted_labels_disaster_response = best_disaster_response_model.predict({'input_ids': tokenized_disaster_response['input_ids'], 'attention_mask': tokenized_disaster_response['attention_mask']})
            
            disaster_response_predictions = []
            for threshold in thresholds:
                disaster_response_prediction = " ".join([labels_disaster_response[i] for i, value in enumerate(predicted_labels_disaster_response[0]) if value > threshold])
                disaster_response_predictions.append(f"Threshold {threshold}: {disaster_response_prediction}")
            
            relevant = "**Relevant**" if timeframe_prediction == "Response" and transfer_prediction == "Request" else "**Not Relevant**"
            
            results.append(f"Text: {text}\nTimeframe: {timeframe_prediction}\nTransfer: {transfer_prediction}\nDisaster Response:\n{'; '.join(disaster_response_predictions)}\nRelevance: {relevant}")
        else:
            results.append("No input provided.")
    
    return "\n\n".join(results)

# Ensure there are 10 text boxes, filling missing entries with empty strings
num_text_boxes = 10
default_texts += [""] * (num_text_boxes - len(default_texts))

# Update Gradio interface to have 10 separate textboxes with default texts
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Textbox(label=f"Text {i+1}", value=default_texts[i], placeholder="Enter your text here...") for i in range(num_text_boxes)],
    outputs=gr.Textbox(label="Predictions"),
    title="Disaster Response Prediction",
    description="Enter texts related to a disaster in each textbox and get predictions for timeframe, transfer type, and actionable labels."
)

if __name__ == "__main__":
    iface.launch()
