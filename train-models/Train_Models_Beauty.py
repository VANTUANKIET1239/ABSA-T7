import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import tempfile

# ==========================================
# 1. CONFIG
# ==========================================
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 2e-5
PATIENCE = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "Models", "Cleanser")
HISTORY_DIR = os.path.join(BASE_DIR, "Histories", "Cleanser")
LOSS_CURVE_DIR = os.path.join(BASE_DIR, "loss_curve", "cleanser")

ASPECTS = [
    'CleansingEffectiveness',
    'HydrationSkinComfort',
    'SkinCompatibilitySafety',
    'TextureFoaming',
    'Fragrance',
    'IngredientsFormulation',
    'Packaging',
    'PriceValue',
    'ShopService',
    'Delivery',
    'General'
]

# Keep the original class indices; None is the internal absent-aspect class.
# Existing targets keep the last sentiment when an aspect has several labels.
POLARITIES = ['positive', 'negative', 'neutral', 'None']  # None index = 3

NUM_ASPECTS = len(ASPECTS)
NUM_POLARITIES = len(POLARITIES)

MODEL_NAME = "vinai/phobert-base"


# ==========================================
# 2. PARSE LABELS
# ==========================================
def parse_labels(raw_label_string):
    if pd.isna(raw_label_string) or not isinstance(raw_label_string, str):
        return []
    parts = raw_label_string.split(';')
    cleaned_labels = [p.strip('{} ') for p in parts if p.strip('{} ')]
    return cleaned_labels


# ==========================================
# 3. CLASS WEIGHTS (WCE)
# ==========================================
def get_class_weights(train_labels_list, device='cpu'):
    all_targets = []
    for labels in train_labels_list:
        target_matrix = np.full((NUM_ASPECTS,), 3, dtype=int)  # default = None
        for label in labels:
            if '#' in label:
                parts = label.split('#')
                if len(parts) == 2:
                    aspect, polarity = parts
                    if aspect in ASPECTS and polarity in POLARITIES:
                        aspect_idx = ASPECTS.index(aspect)
                        polarity_idx = POLARITIES.index(polarity)
                        target_matrix[aspect_idx] = polarity_idx
        all_targets.extend(target_matrix)

    all_targets = np.array(all_targets)

    mask = all_targets != 3
    filtered_targets = all_targets[mask]

    if len(filtered_targets) == 0:
        weights = np.ones(NUM_POLARITIES)
        return torch.tensor(weights, dtype=torch.float32).to(device)

    classes = np.array([0, 1, 2])  # Positive, Negative, Neutral
    raw_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=filtered_targets
    )

    smoothed_weights = np.sqrt(raw_weights)
    clipped_weights = np.clip(smoothed_weights, a_min=0.5, a_max=5.0)

    final_weights = np.ones(NUM_POLARITIES)
    final_weights[0:3] = clipped_weights
    final_weights[3] = 1.0  # None

    print("==== CLASS WEIGHTS (WCE) ====")
    print("Raw weights (0,1,2):", raw_weights)
    print("Final weights:", final_weights)

    return torch.tensor(final_weights, dtype=torch.float32).to(device)


# ==========================================
# 4. LOSS FUNCTIONS
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



# ==========================================
# 5. DATASET
# ==========================================
class ABSADataset(Dataset):
    def __init__(self, texts, labels_list, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        labels = self.labels_list[item]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_matrix = torch.full((NUM_ASPECTS,), 3, dtype=torch.long)  # default = None

        for label in labels:
            if '#' in label:
                aspect, polarity = label.split('#')
                if aspect in ASPECTS and polarity in POLARITIES:
                    target_matrix[ASPECTS.index(aspect)] = POLARITIES.index(polarity)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': target_matrix
        }



class PhoBERT_BiRNN(nn.Module):
    def __init__(self, num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES,
                 rnn_hidden=256, dropout=0.3):
        super().__init__()

        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.phobert.config.hidden_size  # 768

        self.birnn = nn.RNN(
            input_size=hidden_size,
            hidden_size=rnn_hidden,
            batch_first=True,
            bidirectional=True,
            nonlinearity="tanh"   # hoặc "relu"
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_hidden * 2, num_aspects * num_polarities)

        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state  # (B, L, 768)

        rnn_out, _ = self.birnn(seq_output)  # (B, L, hidden*2)

        # masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        rnn_out = rnn_out * mask

        sum_out = torch.sum(rnn_out, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        pooled = sum_out / sum_mask
        pooled = self.dropout(pooled)

        logits = self.fc(pooled)
        return logits.view(-1, self.num_aspects, self.num_polarities)


class PhoBERT_MultiTask(nn.Module):
    def __init__(self, num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
        self.hidden_size = 768 * 4
        self.fc = nn.Linear(self.hidden_size, num_aspects * num_polarities)
        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        cls_concat = torch.cat([
            hidden_states[-1][:, 0, :],
            hidden_states[-2][:, 0, :],
            hidden_states[-3][:, 0, :],
            hidden_states[-4][:, 0, :]
        ], dim=-1)

        logits = self.fc(cls_concat)
        return logits.view(-1, self.num_aspects, self.num_polarities)

class PhoBERT_MultiBranch(nn.Module):
    def __init__(self, num_aspects=12, num_polarities=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True)
        self.hidden_size = 768 * 4

        # Theo tác giả: Phân nhánh thành nhiều mô hình con sử dụng C lớp fully connected độc lập [cite: 190]
        # Mỗi nhánh phụ trách 1 khía cạnh, dự đoán 4 cực cảm xúc.
        self.branches = nn.ModuleList([
            nn.Linear(self.hidden_size, num_polarities) for _ in range(num_aspects)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        cls_concat = torch.cat([
            hidden_states[-1][:, 0, :],
            hidden_states[-2][:, 0, :],
            hidden_states[-3][:, 0, :],
            hidden_states[-4][:, 0, :]
        ], dim=-1)

        # Truyền đặc trưng qua từng nhánh một cách độc lập [cite: 190]
        branch_outputs = [branch(cls_concat) for branch in self.branches]

        # Gộp danh sách các tensors lại thành một tensor duy nhất có dạng (batch_size, num_aspects, num_polarities)
        logits = torch.stack(branch_outputs, dim=1)
        return logits

# ==========================================
# 6. MODEL: PhoBERT + BiLSTM
# ==========================================

import torch
import torch.nn as nn
from transformers import AutoModel

MODEL_NAME = "vinai/phobert-base"

ASPECTS = [
    'CleansingEffectiveness', 'HydrationSkinComfort', 'SkinCompatibilitySafety',
    'TextureFoaming', 'Fragrance', 'IngredientsFormulation', 'Packaging',
    'PriceValue', 'ShopService', 'Delivery', 'General'
]
POLARITIES = ['positive', 'negative', 'neutral', 'None']

NUM_ASPECTS = len(ASPECTS)  # 11
NUM_POLARITIES = len(POLARITIES)  # 4




class PhoBERT_BiGRU(nn.Module):

    def __init__(self, num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES,

                 gru_hidden=256, dropout=0.3):

        super().__init__()



        self.phobert = AutoModel.from_pretrained(MODEL_NAME)

        hidden_size = self.phobert.config.hidden_size  # 768



        self.bigru = nn.GRU(

            input_size=hidden_size,

            hidden_size=gru_hidden,

            batch_first=True,

            bidirectional=True

        )



        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(gru_hidden * 2, num_aspects * num_polarities)



        self.num_aspects = num_aspects

        self.num_polarities = num_polarities



    def forward(self, input_ids, attention_mask):

        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)

        seq_output = outputs.last_hidden_state  # (B, L, 768)



        gru_out, _ = self.bigru(seq_output)  # (B, L, hidden*2)



        # masked mean pooling

        mask = attention_mask.unsqueeze(-1).float()

        gru_out = gru_out * mask



        sum_out = torch.sum(gru_out, dim=1)

        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)



        pooled = sum_out / sum_mask

        pooled = self.dropout(pooled)



        logits = self.fc(pooled)

        return logits.view(-1, self.num_aspects, self.num_polarities)




class PhoBERT_Concat4_BiLSTM_MultiTask(nn.Module):
    def __init__(self, num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES, lstm_hidden=256, dropout=0.3):
        super().__init__()

        # Bật output_hidden_states để lấy các lớp ẩn bên trong
        self.phobert = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)

        # Vì nối 4 lớp cuối của PhoBERT (mỗi lớp 768 chiều)
        rnn_input_size = self.phobert.config.hidden_size * 4  # 3072

        self.bilstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Lớp Linear chung cho Multi-task
        self.fc = nn.Linear(lstm_hidden * 2, num_aspects * num_polarities)

        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

        # Lấy 4 lớp ẩn cuối cùng và nối (concatenate) dọc theo chiều feature (dim=-1)
        concat_features = torch.cat([
            hidden_states[-1],
            hidden_states[-2],
            hidden_states[-3],
            hidden_states[-4]
        ], dim=-1)  # Kích thước: (Batch, Length, 3072)

        # Truyền đặc trưng đã nối qua BiLSTM
        lstm_out, _ = self.bilstm(concat_features)  # (Batch, Length, lstm_hidden*2)

        # Masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        lstm_out = lstm_out * mask

        sum_out = torch.sum(lstm_out, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        pooled = sum_out / sum_mask
        pooled = self.dropout(pooled)

        # Multi-task output
        logits = self.fc(pooled)
        return logits.view(-1, self.num_aspects, self.num_polarities)

class PhoBERT_BiLSTM(nn.Module):
    def __init__(self, num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES, lstm_hidden=256, dropout=0.3):
        super().__init__()

        self.phobert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.phobert.config.hidden_size  # 768

        self.bilstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_aspects * num_polarities)

        self.num_aspects = num_aspects
        self.num_polarities = num_polarities

    def forward(self, input_ids, attention_mask):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        seq_output = outputs.last_hidden_state  # (B, L, 768)

        lstm_out, _ = self.bilstm(seq_output)  # (B, L, hidden*2)

        # masked mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        lstm_out = lstm_out * mask

        sum_out = torch.sum(lstm_out, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)

        pooled = sum_out / sum_mask
        pooled = self.dropout(pooled)

        logits = self.fc(pooled)
        return logits.view(-1, self.num_aspects, self.num_polarities)


# ==========================================
# 7. EVALUATION (SAME AS YOUR CODE)
# ==========================================
def evaluate_model(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            batch_losses = torch.stack([
                loss_fn(logits[:, i, :], targets[:, i])
                for i in range(NUM_ASPECTS)
            ])
            batch_loss = batch_losses.mean()
            total_loss += batch_loss.item()

            preds = torch.argmax(logits, dim=2)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    val_loss = total_loss / len(dataloader)

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    preds_flat = preds_all.flatten()
    targets_flat = targets_all.flatten()

    # ACD
    pred_aspect = (preds_flat != 3).astype(int)
    target_aspect = (targets_flat != 3).astype(int)

    _, _, acd_f1, _ = precision_recall_fscore_support(
        target_aspect, pred_aspect,
        average="macro", zero_division=0
    )

    # SPC (only where target != none)
    mask_spc = (targets_flat != 3)
    preds_spc = preds_flat[mask_spc]
    targets_spc = targets_flat[mask_spc]

    if len(targets_spc) > 0:
        _, _, sent_f1, _ = precision_recall_fscore_support(
            targets_spc, preds_spc,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        )
    else:
        sent_f1 = 0.0

    # ACD + SPC (11*3 multi-label)
    num_samples = targets_all.shape[0]
    y_true_bin = np.zeros((num_samples, NUM_ASPECTS * 3), dtype=int)
    y_pred_bin = np.zeros((num_samples, NUM_ASPECTS * 3), dtype=int)

    for i in range(num_samples):
        for j in range(NUM_ASPECTS):
            t = targets_all[i, j]
            p = preds_all[i, j]
            if t != 3:
                y_true_bin[i, j * 3 + t] = 1
            if p != 3:
                y_pred_bin[i, j * 3 + p] = 1

    _, _, asp_pol_f1, _ = precision_recall_fscore_support(
        y_true_bin, y_pred_bin,
        average="macro", zero_division=0
    )

    return val_loss, acd_f1, sent_f1, asp_pol_f1


# ==========================================
# 8. TRAINING
# ==========================================
def save_checkpoint_safely(state_dict, save_path):
    """Complete the new checkpoint before replacing the previous best weights.

    If Windows blocks replacement of a mapped/open file, keep the completed
    checkpoint under its unique filename and return that actual save path.
    """
    destination = os.path.abspath(os.fspath(save_path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, staged_path = tempfile.mkstemp(
        prefix="cleanser_checkpoint_", suffix=".pt", dir=directory
    )
    keep_staged = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state_dict, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(staged_path, destination)
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33, 1224):
                raise
            keep_staged = True
            print(f"[!] Windows blocked checkpoint replacement: {destination}")
            print(f"[!] New best weights saved instead to: {staged_path}")
            return staged_path
        return destination
    finally:
        if not keep_staged and os.path.exists(staged_path):
            os.remove(staged_path)


def train_model(model, train_loader, val_loader, labels_train=None, epochs=EPOCHS, loss_type='baseline'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    # loss function setup
    if loss_type == 'baseline':
        loss_fn = nn.CrossEntropyLoss()

    elif loss_type == 'class_weights':
        class_weights = get_class_weights(labels_train, device=device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)


    elif loss_type == 'focal_loss':

        # class_weights = get_class_weights(labels_train, device=device)

        # loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

        class_weights = get_class_weights(labels_train, device=device)

        loss_fn = FocalLoss(

            gamma=2,

            alpha=class_weights,

            task_type="multi-class",

            num_classes=NUM_POLARITIES

        )

    else:
        raise ValueError("loss_type không hợp lệ!")

    best_val_joint_f1 = float('-inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    save_path = os.path.join(MODEL_DIR, f'best_model_birnn_cleanser_{loss_type}_VI.pt')
    csv_filename = os.path.join(HISTORY_DIR, f'training_history_birnn_cleanser_{loss_type}_VI.csv')

    metrics_history = []
    print(f"\n[{loss_type.upper()}] Training on device: {device}")
    print("Selecting best checkpoint by validation joint F1; lower validation loss breaks ties.")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")

        for batch in train_loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            aspect_losses = torch.stack([
                loss_fn(logits[:, i, :], targets[:, i])
                for i in range(NUM_ASPECTS)
            ])
            loss = aspect_losses.mean()

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_acd_f1, val_sent_f1, val_asp_pol_f1 = evaluate_model(model, val_loader, device, loss_fn)

        print(f"\n[EPOCH {epoch + 1}]")
        print(f"  Loss -> Train: {avg_train_loss:.4f} | Val: {val_loss:.4f}")
        print(f"  F1   -> ACD: {val_acd_f1:.4f} | SPC: {val_sent_f1:.4f} | ACD+SPC: {val_asp_pol_f1:.4f}")

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'val_acd_f1': val_acd_f1,
            'val_spc_f1': val_sent_f1,
            'val_joint_f1': val_asp_pol_f1
        })
        pd.DataFrame(metrics_history).to_csv(csv_filename, index=False)

        if (val_asp_pol_f1 > best_val_joint_f1 or
                (val_asp_pol_f1 == best_val_joint_f1 and val_loss < best_val_loss)):
            best_val_joint_f1 = val_asp_pol_f1
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_path = save_checkpoint_safely(model.state_dict(), save_path)
            print(f"  => Saved best model: {save_path} "
                  f"(joint val F1={best_val_joint_f1:.4f}, val loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[!] Early stopping at epoch {epoch + 1}")
                break

    df_history = pd.DataFrame(metrics_history)

    return df_history, save_path


# ==========================================
# 9. MAIN
# ==========================================
if __name__ == "__main__":
    print("Loading data...")

    try:
        df_train = pd.read_csv(os.path.join(DATA_DIR, "train_cleanser.csv"), encoding="utf-8")
        df_val = pd.read_csv(os.path.join(DATA_DIR, "val_cleanser.csv"), encoding="utf-8")
        df_test = pd.read_csv(os.path.join(DATA_DIR, "test_cleanser.csv"), encoding="utf-8")

        df_train['parsed_labels'] = df_train['label'].apply(parse_labels)
        df_val['parsed_labels'] = df_val['label'].apply(parse_labels)
        df_test['parsed_labels'] = df_test['label'].apply(parse_labels)

        texts_train = df_train['cleaned_comment'].astype(str).tolist()
        labels_train = df_train['parsed_labels'].tolist()

        texts_val = df_val['cleaned_comment'].astype(str).tolist()
        labels_val = df_val['parsed_labels'].tolist()

        texts_test = df_test['cleaned_comment'].astype(str).tolist()
        labels_test = df_test['parsed_labels'].tolist()

    except Exception as e:
        print(f"[!] Load data error: {e}")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_loader = DataLoader(ABSADataset(texts_train, labels_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ABSADataset(texts_val, labels_val, tokenizer), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(ABSADataset(texts_test, labels_test, tokenizer), batch_size=BATCH_SIZE, shuffle=False)

    SCENARIOS = ['focal_loss','baseline', 'class_weights']
    all_scenario_results = {}

    for scenario in SCENARIOS:
        print("\n" + "=" * 50)
        print(f"RUN SCENARIO: {scenario.upper()}")
        print("=" * 50)

        model = PhoBERT_BiRNN(num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES)

        history_df, saved_model_path = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            labels_train=labels_train,
            epochs=EPOCHS,
            loss_type=scenario
        )

        all_scenario_results[scenario] = history_df

        # plot loss curve
        os.makedirs(LOSS_CURVE_DIR, exist_ok=True)
        plt.figure(figsize=(8, 5))
        plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
        plt.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='o')
        plt.title(f"Loss Curve - Cleanser ABSA - Scenario: {scenario.upper()}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(history_df['epoch'].astype(int))
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(LOSS_CURVE_DIR, f"loss_birnn_cleanser_{scenario}_VI.png"), dpi=300, bbox_inches="tight")
        plt.close()

