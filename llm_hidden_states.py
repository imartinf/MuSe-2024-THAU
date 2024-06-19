import datetime
import json
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_FOLDER = "/autofs/thau03a/datasets/c1_muse_perception"
FACES_FOLDER = os.path.join(DATA_FOLDER, "raw","faces")
LABELS_PATH = os.path.join(DATA_FOLDER, "labels.csv")
PARTITIONS_PATH = os.path.join(DATA_FOLDER, "metadata", "partition.csv")

TEXT_PROMPT_BEG = "The following images show the face of a CEO that is giving a speech.\n"
TEXT_PROMPT_END = "The goal of the task is to characterize the social perception of the CEO based on the images, according to 16 variables related to the Dual Perspective Model: aggressiveness, arrogance, assertiveness, confidence, dominance, independence, leadership qualities, and risk-taking propensity (pertaining to agency), and like collaboration, enthusiasm, friendliness, good-naturedness, kindness, likability, sincerity, and warmth (associated with communality)."
SAMPLE_RATIO = 16
INCLUDE_LABELS = False

def prepare_prompts(subj_id, labels, faces_folder, sample_ratio, text_prompt_beg, text_prompt_end):
    faces_paths = os.listdir(os.path.join(faces_folder, str(subj_id)))
    # Sort the faces by their number
    faces_paths.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    # Skip sample_ratio of the faces
    faces_paths = faces_paths[::sample_ratio]
    subj_labels = labels[labels["subj_id"] == subj_id].drop(columns=["subj_id"])
    if INCLUDE_LABELS:
        text_prompt_end = text_prompt_end.join([f"{k}: {v}" for k, v in subj_labels.iloc[0].items()]) + "."
    prompt_list = [
        {"text": text_prompt_beg},
    ]
    for face_path in faces_paths:
        face_path = os.path.join(faces_folder, str(subj_id), face_path)
        prompt_list.append({"image": face_path})
    prompt_list.append({"text": text_prompt_end})
    return prompt_list


def get_hidden_states(model, tokenizer, prompt_list, save_path):
    query = tokenizer.from_list_format(prompt_list)
    inputs = tokenizer(query, return_tensors='pt')
    inputs = inputs.to(model.device)
    result = model.generate(**inputs, output_hidden_states=True, return_dict_in_generate=True, max_new_tokens=1)
    all_hidden = torch.stack(result["hidden_states"][0], dim=0).squeeze(1)
    torch.save(all_hidden, save_path)

def main():

    labels = pd.read_csv(LABELS_PATH)
    partitions = pd.read_csv(PARTITIONS_PATH)

    model_path = "/autofs/thau00a/home/sestebanro/thau01/models/Qwen-VL-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="cuda:0")


    save_folder = "/mnt/thau03a/datasets/c1_muse_perception/hidden_states/2024-06-13_10-48-55"
    # save_folder = os.path.join(DATA_FOLDER, "hidden_states", datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(save_folder, exist_ok=True)

    # On a config.json file store the used model, sample ratio and prompt

    config = {
        "model_path": model_path,
        "sample_ratio": SAMPLE_RATIO,
        "text_prompt_beg": TEXT_PROMPT_BEG,
        "text_prompt_end": TEXT_PROMPT_END,
        "include_labels": INCLUDE_LABELS,
    }

    # Save the config file
    with open(os.path.join(save_folder, "config.json"), "w") as f:
        json.dump(config, f)

    test_labels = labels[labels["subj_id"].isin(partitions[partitions["Partition"] == "test"]["Id"])]

    # for subj_id in tqdm(labels.dropna()["subj_id"].unique(), total=len(labels.dropna()["subj_id"].unique()), desc="Subjects"):
    for subj_id in tqdm(test_labels["subj_id"], total=len(test_labels["subj_id"]), desc="Test Subjects"):
        prompt_list = prepare_prompts(subj_id, labels, FACES_FOLDER, SAMPLE_RATIO, TEXT_PROMPT_BEG, TEXT_PROMPT_END)
        save_path = os.path.join(save_folder, f"{subj_id}.pt")
        get_hidden_states(model, tokenizer, prompt_list, save_path)

if __name__ == "__main__":
    main()




