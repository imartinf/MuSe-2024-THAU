import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
import wandb

DATA_FOLDER = "/autofs/thau03a/datasets/c1_muse_perception"

FEATURE_NAME_TRAIN = "hidden_states/2024-06-14_15-49-48"
FEATURE_NAME_DEVEL = "hidden_states/2024-06-14_15-49-48"
if "hidden_states" in FEATURE_NAME_TRAIN or "hidden_states" in FEATURE_NAME_DEVEL:
    import torch

LAYER_TO_EXTRACT = -1

VAR2DIM = {
    "aggressive": "agentic",
    "arrogant": "agentic",
    "attractive": "unknown",
    "charismatic": "unknown",
    "competitive": "unknown",
    "dominant": "agentic",
    "enthusiastic": "communal",
    "expressive": "unknown",
    "friendly": "communal",
    "leader_like": "agentic",
    "likeable": "communal",
    "naive": "unknown",
    "assertiv": "agentic",
    "confident": "agentic",
    "independent": "agentic",
    "risk": "agentic",
    "sincere": "communal",
    "collaborative": "communal",
    "kind": "communal",
    "warm": "communal",
    "good_natured": "communal",
}

def scatter_plot_preds_pca(preds, devel_labels, pca):
    # Plot the predicted points and the actual points on the PCA plot using different colors
    plt.figure(figsize=(10, 10))
    plt.scatter(preds['PC1'], preds['PC2'], alpha=0.5, label='Predicted Data')
    plt.scatter(devel_labels['PC1'], devel_labels['PC1'], alpha=0.5, label='Actual Data')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title('PCA Plot with Predicted and Actual Data Points')
    plt.show()

def get_pca_labels(train_labels, devel_labels, pca):

    train_labels_pca = pca.transform(train_labels.drop(columns=["subj_id"]))
    devel_labels_pca = pca.transform(devel_labels.drop(columns=["subj_id"]))

    train_labels_pca = pd.DataFrame(train_labels_pca, columns=["PC1", "PC2"])
    train_labels_pca["subj_id"] = train_labels["subj_id"].values

    devel_labels_pca = pd.DataFrame(devel_labels_pca, columns=["PC1", "PC2"])
    devel_labels_pca["subj_id"] = devel_labels["subj_id"].values

    return train_labels_pca, devel_labels_pca

def load_dataset(feature_name, subj_id, **kwargs):
    subj_data = pd.read_csv(f"{DATA_FOLDER}/feature_segments/{feature_name}/{subj_id}.csv")
    # Return the mean feature for the subject
    return subj_data.drop(columns=["timestamp", "subj_id"]).mean()

def load_dataset_llm(hidden_states_folder, subj_id, **kwargs):
    layer_to_extract = kwargs.get("layer_to_extract", -1)
    hidden_states = torch.load(f"{DATA_FOLDER}/{hidden_states_folder}/{subj_id}.pt")
    hidden_states = hidden_states[layer_to_extract, -1, :].cpu().detach().float().numpy()
    return pd.Series(hidden_states)

def main():

    wandb.init()
    if wandb.config.layer_to_extract:
        LAYER_TO_EXTRACT = wandb.config.layer_to_extract


    if 'hidden_states' in FEATURE_NAME_TRAIN or 'hidden_states' in FEATURE_NAME_DEVEL:
        load_fn = load_dataset_llm
        kwargs = {"layer_to_extract": LAYER_TO_EXTRACT}
    else:
        load_fn = load_dataset
        kwargs = {}

    labels = pd.read_csv(f"{DATA_FOLDER}/labels.csv")

    partitions = pd.read_csv(f"{DATA_FOLDER}/metadata/partition.csv")

    train_labels = labels[labels["subj_id"].isin(partitions[partitions["Partition"] == "train"]["Id"])]
    devel_labels = labels[labels["subj_id"].isin(partitions[partitions["Partition"] == "devel"]["Id"])]

    pca = PCA(n_components=2)
    pca.fit(train_labels.drop(columns=["subj_id"]))

    train_labels_pca, devel_labels_pca = get_pca_labels(train_labels, devel_labels, pca)

    train_features = pd.DataFrame()

    for subj_id in tqdm(train_labels_pca["subj_id"]):
        subj_features = load_fn(FEATURE_NAME_TRAIN, subj_id, **kwargs)
        # Append the new subject feature to the dataset, with subj_id as index
        train_features = pd.concat([train_features, subj_features], axis=1)

    train_features = train_features.T
    train_features.index = train_labels_pca["subj_id"]

    devel_features = pd.DataFrame()

    for subj_id in tqdm(devel_labels_pca["subj_id"]):
        subj_features = load_fn(FEATURE_NAME_DEVEL, subj_id, **kwargs)
        # Append the new subject feature to the dataset, with subj_id as index
        devel_features = pd.concat([devel_features, subj_features], axis=1)

    devel_features = devel_features.T
    devel_features.index = devel_labels_pca["subj_id"]

    # Train a linear regression model
    reg = LinearRegression().fit(train_features, train_labels_pca[["PC1", "PC2"]])

    # Predict the PCA values for the development set
    devel_pred_pca = reg.predict(devel_features)

    results = {}

    # Compute the mean squared error
    mse = mean_squared_error(devel_labels_pca[["PC1", "PC2"]], devel_pred_pca)
    results["mse_pca"] = mse

    # Compute the R2 score
    r2 = r2_score(devel_labels_pca[["PC1", "PC2"]], devel_pred_pca)
    results["r2_pca"] = r2

    # Compute the correlation between the predicted and the true PCA values
    corr_pc1 = pearsonr(devel_labels_pca["PC1"], devel_pred_pca[:, 0])[0]
    results["corr_pc1"] = corr_pc1

    corr_pc2 = pearsonr(devel_labels_pca["PC2"], devel_pred_pca[:, 1])[0]
    results["corr_pc2"] = corr_pc2

    print([f"{k}: {v}" for k, v in results.items()])

    # Plot the predicted points and the actual points on the PCA plot using different colors
    # scatter_plot_preds_pca(pd.DataFrame(devel_pred_pca, columns=["PC1", "PC2"]), devel_labels_pca, pca)

    devel_pred = pca.inverse_transform(devel_pred_pca)

    skip_columns = [k for k, v in VAR2DIM.items() if v == "unknown"]

    results_per_label = {}

    for i, var in enumerate(devel_labels.columns[1:]):
        if var in skip_columns:
            continue
        mse = mean_squared_error(devel_labels[var], devel_pred[:, i])
        r2 = r2_score(devel_labels[var], devel_pred[:, i])
        corr = pearsonr(devel_labels[var], devel_pred[:, i])[0]
        results_per_label[var] = {"mse": mse, "r2": r2, "corr": corr}

    print(results_per_label)
    mean_results = {}
    mean_results["mse"] = np.mean([v["mse"] for v in results_per_label.values()])
    mean_results["r2"] = np.mean([v["r2"] for v in results_per_label.values()])
    mean_results["corr"] = np.mean([v["corr"] for v in results_per_label.values()])
    print(mean_results)

    # Log mean pearson correlation
    wandb.log({"mean_corr": mean_results["corr"]})

if __name__ == "__main__":
    sweep_config = {
        "method": "grid",
        "parameters": {
            "layer_to_extract": {
                "values": [-34]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="lr_on_pca")

    wandb.agent(sweep_id, function=main)