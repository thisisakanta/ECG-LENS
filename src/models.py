import torch
import ResNet1D
import numpy as np
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch, caps=None, caplens=None, teacher_forcing=True, token_encoder=None, max_caption_len=None):
        x = self.encoder(batch)
        y = self.decoder(x, caps, caplens, teacher_forcing, token_encoder, max_caption_len)

        return y

class Encoder(torch.nn.Module):
    """
    Multi-Lead Encoder - outputs configurable per-lead embeddings + configurable global embeddings.
    Supports both standard ResNet1D and SSL pre-trained models.
    """

    def __init__(self, num_blocks, encoded_ecg_size, kernel_size, stem_kernel_size, in_chan, 
                 ssl_pretrained_path=None, ssl_ecg_model=None, encoder_dim=512,
                 num_leads=12, use_lead_embeddings=True, tokens_per_lead=1):
        """
        Args:
            num_blocks: Number of ResNet blocks (18 or 34)
            encoded_ecg_size: Unused (kept for backward compat)
            kernel_size: Kernel size for ResNet
            stem_kernel_size: Stem kernel size for ResNet
            in_chan: Number of input channels (typically 12 for ECG leads)
            ssl_pretrained_path: Path to SSL pre-trained checkpoint (optional)
            ssl_ecg_model: SSL ECG model type (e.g., 'resnet50', 'vit_tiny_ecg')
            encoder_dim: Encoder output embedding dimension
            num_leads: Number of ECG leads (typically 12)
            use_lead_embeddings: Whether to add learnable lead identity embeddings
            tokens_per_lead: Number of learned context tokens per lead
        """
        super(Encoder, self).__init__()
        
        self.use_ssl = ssl_pretrained_path is not None
        self.encoded_ecg_size = encoded_ecg_size  # kept for backward compat
        self.encoder_dim = encoder_dim
        self.num_leads = num_leads
        self.tokens_per_lead = tokens_per_lead
        self.global_tokens = tokens_per_lead
        self.num_ecg_tokens = num_leads * tokens_per_lead + self.global_tokens
        self.use_lead_embeddings = use_lead_embeddings

        if tokens_per_lead < 1:
            raise ValueError(f"tokens_per_lead must be >= 1, got {tokens_per_lead}")

        if self.tokens_per_lead > 1:
            # Used by SSL mode, where we only have one global embedding and expand it to K tokens.
            self.lead_token_expander = nn.Linear(encoder_dim, encoder_dim * self.tokens_per_lead)
            self.intra_lead_token_embeddings = nn.Embedding(self.tokens_per_lead, encoder_dim)
            self.global_token_expander = nn.Linear(encoder_dim, encoder_dim * self.global_tokens)
            self.intra_global_token_embeddings = nn.Embedding(self.global_tokens, encoder_dim)
        
        # Embedding for lead identity (e.g., "this is lead I", "this is lead II", etc.)
        if use_lead_embeddings:
            self.lead_embeddings = nn.Embedding(num_leads + 1, encoder_dim)  # +1 for global
            self.modality_embedding = nn.Embedding(2, encoder_dim)  # 0 for ECG, 1 reserved
        
        if self.use_ssl:
            # SSL pre-trained encoder path
            print(f"Loading SSL pre-trained encoder from {ssl_pretrained_path}")
            from utils_builder import ECGCLIP
            
            network_config = {
                "projection_head": {
                    "mlp_hidden_size": 256,
                    "projection_size": 256,
                },
                "ecg_model": ssl_ecg_model,
                "num_leads": 12,
                "text_model": "ncbi/MedCPT-Query-Encoder",
            }
            
            ssl_model = ECGCLIP(network_config)
            ckpt = torch.load(ssl_pretrained_path, map_location="cpu")
            ssl_model.load_state_dict(ckpt, strict=True)
            
            self.ecg_encoder = ssl_model.ecg_encoder
            self.ssl_model = ssl_model
            self.ssl_ecg_model = ssl_ecg_model
            self.ssl_model.eval()
            self.ssl_proj_dim = ssl_model.proj_out
            self.ssl_projection = nn.Linear(self.ssl_proj_dim, encoder_dim)
            
            print(f"Loaded SSL ECGCLIP encoder (model: {ssl_ecg_model})")
            print(f"SSL projection: {self.ssl_proj_dim} -> {encoder_dim}")
            
        else:
            # Standard ResNet1D encoder path - create per-lead encoders
            self.per_lead_encoders = nn.ModuleList()
            for lead_idx in range(num_leads):
                if num_blocks == 18:
                    resnet = ResNet1D.resnet18(encoder_dim, kernel_size=kernel_size, 
                                              stem_kernel_size=stem_kernel_size, in_chan=1)
                elif num_blocks == 34:
                    resnet = ResNet1D.resnet34(encoder_dim, kernel_size=kernel_size,
                                              stem_kernel_size=stem_kernel_size, in_chan=1)
                else:
                    raise ValueError(f"num_blocks must be 18 or 34, got {num_blocks}")
                
                # Modify to output encoder_dim features
                resnet.avgpool = torch.nn.AdaptiveAvgPool1d(1)
                self.per_lead_encoders.append(resnet)
            
            self.embedding_size = encoder_dim
            
            # Global ResNet encoder that takes all 12 channels as input
            if num_blocks == 18:
                self.global_encoder = ResNet1D.resnet18(encoder_dim, kernel_size=kernel_size, 
                                                        stem_kernel_size=stem_kernel_size, in_chan=num_leads)
            elif num_blocks == 34:
                self.global_encoder = ResNet1D.resnet34(encoder_dim, kernel_size=kernel_size,
                                                        stem_kernel_size=stem_kernel_size, in_chan=num_leads)
            
            self.global_encoder.avgpool = torch.nn.AdaptiveAvgPool1d(1)

            # Temporal token extraction directly from ResNet feature maps.
            self.temporal_pool = nn.AdaptiveAvgPool1d(self.tokens_per_lead)
            backbone_dim = self.global_encoder.fc.in_features
            self.lead_temporal_projection = nn.Linear(backbone_dim, encoder_dim)
            self.global_temporal_projection = nn.Linear(backbone_dim, encoder_dim)

        # LayerNorm for all ECG embeddings
        self.ecg_norm = nn.LayerNorm(encoder_dim)

        print("number of Encoder parameters: %.2fM" % (self.get_num_params() / 1e6,))
        
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent gradient computation for encoder.
        
        :param fine_tune: Allow fine-tuning?
        """
        for p in self.parameters():
            p.requires_grad = fine_tune

    @staticmethod
    def _forward_resnet_feature_map(resnet_model, signal_batch):
        """Run ResNet1D up to the last temporal feature map (before avgpool/fc)."""
        x = resnet_model.conv1(signal_batch)
        x = resnet_model.maxpool(x)
        x = resnet_model.stages(x)
        x = resnet_model.bn1(x)
        x = resnet_model.relu1(x)
        return x

    def _extract_temporal_tokens(self, resnet_model, signal_batch, projection_layer):
        """Extract K temporal tokens via adaptive pooling over ResNet feature maps."""
        feat_map = self._forward_resnet_feature_map(resnet_model, signal_batch)  # (B, C, T)
        pooled = self.temporal_pool(feat_map)  # (B, C, K)
        tokens = pooled.transpose(1, 2).contiguous()  # (B, K, C)
        return projection_layer(tokens)  # (B, K, D)

    def forward(self, batch):
        """
        Forward propagation.
        
        :param batch: ECG signals, tensor of shape (batch_size, num_leads, signal_length)
        :return: encoded ECG, tensor of shape (batch_size, num_leads*tokens_per_lead + global_tokens, encoder_dim)
        """
        batch_size = batch.size(0)
        device = batch.device
        
        if self.use_ssl:
            # SSL encoder path - outputs single global embedding
            with torch.no_grad():
                if "resnet" in self.ssl_ecg_model:
                    x = self.ecg_encoder(batch)
                    x = self.ssl_model.downconv(x)
                    x, _ = self.ssl_model.att_pool_head(x)
                    ssl_emb = x.view(x.shape[0], -1)
                elif "vit_tiny" in self.ssl_ecg_model:
                    x = self.ecg_encoder(batch)
                    ssl_emb = self.ssl_model.proj_e(x)
            
            proj_emb = self.ssl_projection(ssl_emb)  # (batch_size, encoder_dim)
            global_emb = proj_emb.unsqueeze(1)  # (batch_size, 1, encoder_dim)
            
            # For SSL, we replicate global embedding to all 12 leads for now
            # (can be improved with per-lead SSL encoders if available)
            lead_embs = global_emb.expand(batch_size, self.num_leads, self.encoder_dim)
            
        else:
            # Standard ResNet1D - extract temporal tokens directly from feature maps.
            per_lead_tokens = []
            for lead_idx in range(self.num_leads):
                lead_signal = batch[:, lead_idx:lead_idx + 1, :]  # (batch_size, 1, signal_length)
                lead_tokens = self._extract_temporal_tokens(
                    self.per_lead_encoders[lead_idx],
                    lead_signal,
                    self.lead_temporal_projection,
                )  # (batch_size, tokens_per_lead, encoder_dim)
                per_lead_tokens.append(lead_tokens)

            lead_embs = torch.cat(per_lead_tokens, dim=1)  # (batch_size, num_leads*tokens_per_lead, encoder_dim)
            global_emb = self._extract_temporal_tokens(
                self.global_encoder,
                batch,
                self.global_temporal_projection,
            )  # (batch_size, global_tokens, encoder_dim)

        # Apply LayerNorm to per-lead embeddings.
        lead_embs = self.ecg_norm(lead_embs)

        if self.use_ssl:
            # SSL path still starts from one vector; expand to K tokens when configured.
            if self.tokens_per_lead > 1:
                lead_embs = self.lead_token_expander(lead_embs)
                lead_embs = lead_embs.view(batch_size, self.num_leads, self.tokens_per_lead, self.encoder_dim)
                token_ids = torch.arange(self.tokens_per_lead, device=device)
                token_bias = self.intra_lead_token_embeddings(token_ids).view(
                    1, 1, self.tokens_per_lead, self.encoder_dim
                )
                lead_embs = lead_embs + token_bias
                lead_embs = lead_embs.reshape(batch_size, self.num_leads * self.tokens_per_lead, self.encoder_dim)
                lead_indices = torch.arange(self.num_leads, device=device).repeat_interleave(self.tokens_per_lead)
                lead_indices = lead_indices.unsqueeze(0).expand(batch_size, -1)
            else:
                lead_indices = torch.arange(self.num_leads, device=device).unsqueeze(0).expand(batch_size, -1)

            global_emb = self.ecg_norm(global_emb)
            if self.global_tokens > 1:
                global_emb = self.global_token_expander(global_emb.squeeze(1))
                global_emb = global_emb.view(batch_size, self.global_tokens, self.encoder_dim)
                global_token_ids = torch.arange(self.global_tokens, device=device)
                global_token_bias = self.intra_global_token_embeddings(global_token_ids).view(
                    1, self.global_tokens, self.encoder_dim
                )
                global_emb = global_emb + global_token_bias
        else:
            lead_indices = torch.arange(self.num_leads, device=device).repeat_interleave(self.tokens_per_lead)
            lead_indices = lead_indices.unsqueeze(0).expand(batch_size, -1)
            global_emb = self.ecg_norm(global_emb)

        # Add lead identity embeddings if enabled.
        if self.use_lead_embeddings:
            lead_id_embs = self.lead_embeddings(lead_indices)
            modality_embs = self.modality_embedding(torch.zeros_like(lead_indices))
            lead_embs = lead_embs + lead_id_embs + modality_embs * 0.1
        
        if self.use_lead_embeddings:
            global_indices = torch.full(
                (batch_size, self.global_tokens),
                self.num_leads,
                device=device,
                dtype=torch.long,
            )
            global_id_embs = self.lead_embeddings(global_indices)
            global_modality_embs = self.modality_embedding(torch.zeros_like(global_indices))
            global_emb = global_emb + global_id_embs + global_modality_embs * 0.1

        output = torch.cat([lead_embs, global_emb], dim=1)
        
        return output


class Attention(torch.nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = torch.nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = torch.nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = torch.nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, downsampled_signal_length, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, embed_length)
        alpha = self.softmax(att)  # (batch_size, downsampled_signal_length)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(torch.nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, token_encoder, encoder_dim, layers, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = token_encoder.vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = torch.nn.Embedding(self.vocab_size, embed_dim)  # embedding layer
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.decode_step = torch.nn.LSTM(embed_dim + encoder_dim, decoder_dim, num_layers=layers, bias=True)  # decoding LSTMCell
        self.init_h = torch.nn.Linear(encoder_dim, layers * decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = torch.nn.Linear(encoder_dim, layers * decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = torch.nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = torch.nn.Sigmoid()
        self.fc = torch.nn.Linear(decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

        print("number of LSTM-decoder parameters: %.2fM" % (self.get_num_params() / 1e6,))
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = torch.nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1) # later we will aggregate with alphas, for now we do a simple mean
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None, teacher_forcing=True, token_encoder=None,
                max_length=None):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :param teacher_forcing: boolean whether in each step to feed next ground truth or next predicted token
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        device = encoder_out.device

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        if teacher_forcing:
            # Sort input data by decreasing lengths; why? apparent below
            caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
            encoder_out = encoder_out[sort_ind]

            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
            h = h.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()
            c = c.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()
            encoded_captions = encoded_captions[sort_ind]
            # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
            # So, decoding lengths are actual lengths - 1
            decode_lengths = (caption_lengths - 1).tolist()
            # Embedding
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
            # Create tensors to hold word prediction scores and alphas
            predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
            alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
            for t in range(max(decode_lengths)):
                batch_size_t = sum([l > t for l in decode_lengths])
                h_pooled = h.sum(0)[:batch_size_t]
                attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                    h_pooled)  # (batch_size_t, encoder_dim), (batch_size_t, num_pixels)
                gate = self.sigmoid(self.f_beta(h_pooled))  # gating scalar, (batch_size_t, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding
                h_last, (h, c) = self.decode_step(
                    torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1).unsqueeze(0),
                    (h[:, :batch_size_t].contiguous(),
                     c[:, :batch_size_t].contiguous()))  # (batch_size_t, decoder_dim)
                preds = self.fc(self.dropout(h_last.squeeze(0)))  # (batch_size_t, vocab_size)
                predictions[:batch_size_t, t, :] = preds
                alphas[:batch_size_t, t, :] = alpha
        else:
            # Initialize LSTM state
            h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
            h = h.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()
            c = c.reshape(batch_size, -1, self.decoder_dim).permute(1, 0, 2).contiguous()

            # Create tensors to hold word prediction scores and alphas
            predictions = torch.zeros(batch_size, max_length, vocab_size).to(device)
            alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)

            prev_words = torch.LongTensor([[token_encoder.special_tokens['<|startoftext|>']]] * batch_size).to(
                device)
            is_seq_complete = np.full(len(encoder_out), False)
            decode_lengths = torch.ones(len(encoder_out))
            step = 0
            while step < max_length:
                embeddings = self.embedding(prev_words).squeeze(1)  # (s, embed_dim)
                h_pooled = h.sum(0)
                attention_weighted_encoding, alpha = self.attention(encoder_out,
                                                                    h_pooled)  # (s, encoder_dim), (s, num_pixels)
                alpha = alpha.view(-1, encoder_out.size(1))  # (s, enc_image_size, enc_image_size)

                gate = self.sigmoid(self.f_beta(h_pooled))  # gating scalar, (s, encoder_dim)
                attention_weighted_encoding = gate * attention_weighted_encoding

                h_last, (h, c) = self.decode_step(
                    torch.cat([embeddings, attention_weighted_encoding], dim=1).unsqueeze(0),
                    (h, c))  # (s, decoder_dim)
                preds = self.fc(self.dropout(h_last.squeeze(0)))  # (batch_size_t, vocab_size)
                predictions[:, step, :] = preds
                alphas[:, step, :] = alpha

                next_word_inds = torch.argmax(preds, 1)

                # Which sequences are incomplete (didn't reach <end>)?
                cur_incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                       next_word != token_encoder.special_tokens['<|endoftext|>']]
                # all complete sequences (previous + new)
                cur_complete_inds = list(set(range(len(next_word_inds))) - set(cur_incomplete_inds))
                # only the sequences that reached <|end|> in this iteration
                new_complete_inds = list(
                    set(cur_complete_inds) - set(i for i, is_complete in enumerate(is_seq_complete) if is_complete))

                if len(new_complete_inds) > 0:
                    is_seq_complete[new_complete_inds] = True
                    decode_lengths[
                        new_complete_inds] = step + 1  # lengths is the sequence with start but witout end token

                prev_words = next_word_inds.unsqueeze(1)

                if np.all(is_seq_complete):
                    break

                step += 1

            if not np.all(is_seq_complete):
                incomplete_ids = [i for i, is_complete in enumerate(is_seq_complete) if not is_complete]
                decode_lengths[incomplete_ids] = max_length - 1  # -1 because we don't include end token

            decode_lengths, sort_ind = decode_lengths.sort(descending=True)
            if encoded_captions is not None:
                encoded_captions = encoded_captions[sort_ind]
            predictions = predictions[sort_ind]
            alphas = alphas[sort_ind]
            decode_lengths = decode_lengths.int().tolist()
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
