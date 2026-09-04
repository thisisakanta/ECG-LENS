import ast
import matplotlib.pyplot as plt
import pickle
import os
import torch
from my_datasets import CaptionDataset
from train_transformer_decoder import Model
import models
from nltk.translate.bleu_score import corpus_bleu
from utils import word_tokenize_caption_list
import skimage
import numpy as np
import configparser
from collections import Counter
from nltk.translate.meteor_score import meteor_score
from statistics import mean
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu", 0)


class CaptionScriptDataset(Dataset):
    def __init__(self, ds, ids_to_plot):
        """
        Initialize the custom dataset.
        Args:
            ds: The primary dataset object.
            ids_to_plot: An array of IDs to include with each item.
        """
        self.ds = ds
        self.ids_to_plot = ids_to_plot

        assert len(self.ds) == len(self.ids_to_plot), "The number of items in the dataset and the number of IDs must match."

    def __len__(self):
        # Return the minimum length of both sources to avoid out-of-bounds errors
        return len(self.ds)

    def __getitem__(self, idx):
        # Get data from the primary dataset
        data = self.ds[idx]
        # Get the corresponding ID from the ids_to_plot array
        id_to_plot = self.ids_to_plot[idx]
        return data, id_to_plot

def visualize_att(output_folder, ecg_id, ecg, hypotheses, reference, alphas, score, output_format, lead_to_plot,
                  smooth=True):
    """
    Visualizes the caption with attention weights for each word.

    :param output_folder: Directory to save the visualization images
    :param ecg: Input image
    :param hypotheses: Generated captions
    :param reference: Reference caption
    :param alphas: Attention weights
    :param score: BLEU score of the caption
    :param smooth: Flag to apply smoothing to the attention weights
    """
    alphas = alphas.cpu().numpy()
    x = np.linspace(0, 60, len(ecg))
    # x = torch.linspace(0, 60, steps=image.shape[-1], device="cpu")
    y = ecg
    # Calculate the number of rows and columns needed
    n_words = min(len(hypotheses), 50)  # Limit to 50 words as in your original code
    n_cols = min(3, n_words)  # Up to 3 columns
    n_rows = (n_words + n_cols - 1) // n_cols  # Ceiling division to get number of rows

    # Increase the default font size
    plt.rcParams.update({'font.size': 18})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*n_cols, 5*n_rows), sharey=True)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    y_limits = [np.min(y) - 0.1 * abs(np.min(y)), np.max(y) + 0.1 * abs(np.max(y))]

    for i, (cur_alpha, ax) in enumerate(zip(alphas, axes)):
        if i >= 50:
            break
        ax.plot(x, y, label='Original plot', color='blue', linewidth=0.75)

        if smooth:
            alpha = skimage.transform.pyramid_expand(np.array(cur_alpha), upscale=len(ecg) / len(cur_alpha), sigma=8)
        else:
            alpha = skimage.transform.resize(np.array(cur_alpha), [len(ecg)])

        alpha_normed = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha))
        text = hypotheses[i]

        ax.set_ylim(y_limits)
        X, Y = np.meshgrid(x, np.linspace(0, 1, 2))
        cmap = plt.cm.gray
        cmap.set_bad(color='none')

        alpha_grid = np.repeat(alpha_normed[np.newaxis, :], 2, axis=0)
        mesh = ax.pcolormesh(X, Y * (ax.get_ylim()[1] - ax.get_ylim()[0]) + ax.get_ylim()[0],
                             np.ones_like(alpha_grid), cmap=cmap, shading='auto', alpha=1)
        mesh.set_array(None)
        mesh.set_alpha(alpha_grid.flatten())

        ax.set_title(text)
        # Remove all ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # Remove any unused subplots
    for ax in axes[n_words:]:
        ax.remove()

    fig.suptitle("Reference: " + reference, fontsize=24)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.08, wspace=0.3, hspace=0.3)
    
    os.makedirs(output_folder, exist_ok=True)
    if output_format == "pdf":
        plt.savefig(os.path.join(output_folder, f"{ecg_id}_attention_lead_{lead_to_plot}_METEOR={score:.4f}.pdf"),
                format='pdf',
                dpi=300,
                bbox_inches='tight')
    elif output_format == "png":
        plt.savefig(os.path.join(output_folder, f"{ecg_id}_attention_lead_{lead_to_plot}_METEOR={score:.4f}.png"), dpi=150)
    plt.close(fig)  # Close the figure to free up memory



if __name__ == '__main__':

    import __main__
    setattr(__main__, "Model", Model)
    torch.cuda.empty_cache()

    # Read configuration settings
    config = configparser.ConfigParser()
    config.read("configs/config_caption.ini")
    paths = config["files and paths"]
    model_paths = config["MODELS"]

    # Extract settings from the configuration file
    abbreviation_script = config.getboolean("general", "abbreviation_script")
    make_figs = config.getboolean("general", "make_figs")
    visualize_attention = config.getboolean("general", "visualize_attention")
    ecg_length = config.getint("general", "ecg_length")
    min_reference_length_filter = config.getint("general", "min_reference_length_filter")
    max_reference_length_filter = config.getint("general", "max_reference_length_filter")
    max_prediction_length_filter = config.getint("general", "max_prediction_length_filter")
    min_prediction_length_filter = config.getint("general", "min_prediction_length_filter")
    max_num_samples = config.getint("general", "max_num_samples")
    normalize_ds = config.getboolean("general", "normalize_ds")
    batch_size = config.getint("general", "batch_size")
    workers = config.getint("general", "workers")
    lead_to_plot = config.getint("general", "lead_to_plot")
    output_format = config.get("general", "output_format")
    max_caption_length = config.getint("general", "max_caption_length")
    ecg_ids_str = config.get("general", "ecg_ids")

    data_folder = paths['data_folder']
    token_encoder_path = paths['token_encoder_path']
    figs_output_folder = paths['figs_output_folder']
    attention_output_folder = paths['attention_output_folder']

    lstm_model_filename = model_paths['lstm_model_filename']
    gpt_model_filename = model_paths['gpt_model_filename']

    ecg_ids = ast.literal_eval(ecg_ids_str)
    with open(token_encoder_path, 'rb') as f:
        token_encoder = pickle.load(f)

    ds = CaptionDataset(data_folder, 'TEST')

    ecgs = list(ds.secgs)
    captions = list(ds.captions)
    caplens = list(ds.caplens)
    if not ecg_ids:
        ecg_ids = range(len(ds))
    ecgs = np.array(ecgs)
    ecgs = ecgs[ecg_ids]
    captions = [captions[i] for i in ecg_ids]
    caplens = [caplens[i] for i in ecg_ids]
    in_chan = 12

    if max_num_samples != -1:
        ecgs = ecgs[:max_num_samples]
        captions = captions[:max_num_samples]
        caplens = caplens[:max_num_samples]
        ecg_ids = ecg_ids[:max_num_samples]

    ds = CaptionDataset(ecgs=ecgs, captions=captions, caplens=caplens, read_from_hdf5=False)

    if ds.secgs.ndim == 2:
        ds.secgs = np.expand_dims(ds.secgs, 1)
    if normalize_ds:
        ds_mean = ds.secgs.mean(axis=(0, 2), keepdims=True)
        ds_std = ds.secgs.std(axis=(0, 2), keepdims=True)
        ds.secgs = (ds.secgs - ds_mean) / ds_std

    ds_with_ids = CaptionScriptDataset(ds, ecg_ids)
    data_loader = torch.utils.data.DataLoader(
        ds_with_ids, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    # Load the LSTM model checkpoint
    lstm_checkpoint = torch.load(lstm_model_filename, map_location=device)
    gpt_checkpoint = torch.load(gpt_model_filename, map_location=(device))
    #gpt_encoder_config_dict = gpt_checkpoint['encoder_config']
    
    # Get encoder config and add backward compatibility for SSL parameters
    encoder_config = lstm_checkpoint["encoder_config"]
    if 'ssl_pretrained_path' not in encoder_config:
        encoder_config['ssl_pretrained_path'] = None
    if 'ssl_ecg_model' not in encoder_config:
        encoder_config['ssl_ecg_model'] = None
    if 'encoder_dim' not in encoder_config:
        # Default encoder_dim for old checkpoints (standard ResNet output channels)
        encoder_config['encoder_dim'] = 512
    
    lstm_encoder = models.Encoder(**encoder_config).to(device)
    lstm_decoder = models.DecoderWithAttention(**lstm_checkpoint['decoder_config'],
                                               token_encoder=token_encoder).to(device)
    lstm_model = models.Model(lstm_encoder, lstm_decoder).to(device)
    lstm_model.load_state_dict(lstm_checkpoint['model'])
    lstm_model.eval()


    gpt_encoder = models.Encoder(**encoder_config).to(device)
    gpt_decoder_config_dict = gpt_checkpoint['decoder_config']
    gpt_model = Model(gpt_decoder_config_dict, gpt_encoder, token_encoder,
                        encoder_config['encoded_dim'])
    gpt_model.load_state_dict(gpt_checkpoint['model'])
    gpt_model.to(device)
    gpt_model.eval()

    hypothesis_lstm_list = []
    all_alphas = []

    captions_lstm = []
    references = []
    references_abbr = []
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    with torch.no_grad():
        for i, ((batch, captions, caplens), batch_ids) in tqdm(enumerate(data_loader)):
            batch = batch.float().to(device)
            captions = captions.to(device)
            caplens = caplens.to(device)

            # Filter based on reference length
            valid_indices = []
            for j in range(len(batch)):
                min_condition = (min_reference_length_filter <= caplens[j]) if min_reference_length_filter != -1 else True
                max_condition = (caplens[j] <= max_reference_length_filter) if max_reference_length_filter != -1 else True
                if min_condition and max_condition:
                    valid_indices.append(j)

            if not valid_indices:
                continue  # Skip this batch if no valid references

            # Filter batch data
            batch = batch[valid_indices]
            captions = captions[valid_indices]
            caplens = caplens[valid_indices]
            batch_ids = batch_ids[valid_indices]

            # Generate caption using beam search
            scores, caps_sorted, decode_lengths, alphas, sort_ind = lstm_model(batch, caps=None, caplens=None,
                                                                               teacher_forcing=False,
                                                                               token_encoder=token_encoder,
                                                                               max_caption_len=300)
            all_alphas.extend(alphas)
            _, preds_lstm = torch.max(scores, dim=2)
            batch_decoded_hypotheses_lstm = [token_encoder.decode(pred[:decode_lengths[i]], skip_special_tokens=True)
                                             for i, pred in enumerate(preds_lstm)]
            batch_tokenized_lstm_hypotheses = word_tokenize_caption_list(batch_decoded_hypotheses_lstm, exceptions=["<|unk|>"])

            batch = batch[sort_ind]
            captions = captions[sort_ind]
            caplens = caplens[sort_ind]
            batch_ids = batch_ids[sort_ind]

            valid_indices = []
            for j, hypotheses in enumerate(batch_tokenized_lstm_hypotheses):
                hypotheses_len = len(hypotheses)
                min_condition = (min_prediction_length_filter <= hypotheses_len) if min_prediction_length_filter != -1 else True
                max_condition = (hypotheses_len <= max_prediction_length_filter) if max_prediction_length_filter != -1 else True
                if min_condition and max_condition:
                    valid_indices.append(j)

            if not valid_indices:
                continue  # Skip this batch if no valid references

            batch = batch[valid_indices]
            captions = captions[valid_indices]
            caplens = caplens[valid_indices]
            batch_ids = batch_ids[valid_indices]

            batch_tokenized_lstm_hypotheses = [batch_tokenized_lstm_hypotheses[j] for j in valid_indices]

            batch_decoded_references = [token_encoder.decode(cap, skip_special_tokens=True) for cap in captions]
            batch_references = word_tokenize_caption_list(batch_decoded_references)
            references.extend([[ref] for ref in batch_references])
            hypothesis_lstm_list.extend(batch_tokenized_lstm_hypotheses)

            batch_ids = [str(i) for i in batch_ids.tolist()]
            for j, (ecg, lstm_hypothesis) in enumerate(zip(batch, batch_tokenized_lstm_hypotheses)):
                ecg_id = batch_ids[j]
                meteor_lstm = round(meteor_score([batch_references[j]], lstm_hypothesis), 4)
                # Calculate BLEU scores
                bleu4_lstm = corpus_bleu([[batch_references[j]]], [lstm_hypothesis])
                bleu1_lstm = corpus_bleu([[batch_references[j]]], [lstm_hypothesis], weights=[1, 0, 0, 0])

                if visualize_attention:
                    ecg_single_lead = ecg[lead_to_plot-1].cpu().numpy()
                    # Move tensors to CPU and convert to NumPy arrays
                    cur_alpha = alphas[j][:len(lstm_hypothesis)].detach()
                    # Visualize the attention (alphas) of every word, skipping the start token
                    visualize_att(attention_output_folder, ecg_id, ecg_single_lead, batch_tokenized_lstm_hypotheses[j],
                                  batch_decoded_references[j], cur_alpha, meteor_lstm, output_format, leads[lead_to_plot-1],
                                  smooth=True)

                print("Ground Truth is \n", batch_references[j])
                print("LSTM Prediction is \n", lstm_hypothesis)
                print("METEOR-Score is \n", meteor_lstm)


                if make_figs:
                    # Create a figure with ground truth and predicted captions
                    t = 'GROUND TRUTH:\n' \
                        '               %s\n\n' \
                        'PREDICTIONS:\n' \
                        '       LSTM: METEOR-score %s\n ' \
                        '               %s' % (
                                              batch_decoded_references[j],
                                              str(meteor_lstm), batch_decoded_hypotheses_lstm[j])

                    ecg = ecg.cpu().numpy()
                    num_channels = ecg.shape[0]
                    cols = min(num_channels, 4)  # Adjust columns to the minimum of num_channels or 4
                    rows = (num_channels + cols - 1) // cols  # Calculate rows needed for the given number of channels
                    # Dynamic figure size based on number of rows and columns
                    fig_width = 7 * cols
                    fig_height = 2.5 * rows + 2 # Extra space at the bottom for the text
                    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

                    # Flatten axes array if it’s 2D; wrap it in a list if there’s only one subplot
                    if num_channels == 1:
                        axes = [axes]
                    elif isinstance(axes, np.ndarray):
                        axes = axes.flatten()
                    # Adjust spacing
                    fig.subplots_adjust(hspace=0.5, wspace=0.3, bottom=0.2)
                    # Font sizes based on the number of channels (adjust as needed)
                    title_fontsize = max(10, 14 - rows)
                    text_fontsize = max(16, 18 - rows)

                    # Plot each channel
                    for k in range(num_channels):
                        axes[k].plot(ecg[k], linewidth=0.75)
                        axes[k].set_title(f"Lead {leads[k]}", fontsize=title_fontsize)
                        axes[k].set_xticks([])

                    # Hide unused subplots
                    for k in range(num_channels, rows * cols):
                        axes[k].axis('off')

                    # Adding the text at the bottom with adaptive font size
                    fig.text(0.5, 0.05, t, ha='center', va='top', fontsize=text_fontsize,
                             bbox=dict(facecolor='white', alpha=0.8))
                    #
                    os.makedirs(figs_output_folder, exist_ok=True)
                    output_path = os.path.join(figs_output_folder, f"{ecg_id}_METEOR={meteor_lstm:.4f}.{output_format}")
                    #
                    plt.savefig(output_path, format=output_format, dpi=300, bbox_inches='tight')

                    plt.close(fig)


    # Calculate and print the average METEOR and BLEU scores for the entire dataset
    corpus_meteor_lstm = mean(meteor_score(cur_reference, cur_hypothesis) for cur_reference, cur_hypothesis
         in zip(references, hypothesis_lstm_list))
    corpus_bleu_1_lstm = corpus_bleu(references, hypothesis_lstm_list, weights=[1, 0, 0, 0])
    corpus_bleu_4_lstm = corpus_bleu(references, hypothesis_lstm_list)

    print("most common LSTM predicted tokens: ",
          str(Counter([sub_item for item in hypothesis_lstm_list for sub_item in item]).most_common()[:30]))
    print("most common references tokens: ",
          str(Counter([item for sublist1 in references for sublist2 in sublist1 for item in sublist2]).most_common()[:30]))

    print("mean_meteor_lstm is: " + str(corpus_meteor_lstm))
    print("corpus_bleu_4_lstm is: " + str(corpus_bleu_4_lstm))
    print("corpus_bleu_1_lstm is: " + str(corpus_bleu_1_lstm))
