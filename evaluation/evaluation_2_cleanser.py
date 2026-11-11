import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import ast
from tqdm.auto import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix
import os
# ==========================================
# 1. CẤU HÌNH & HẰNG SỐ (CONFIGURATIONS)
# ==========================================
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 2e-4
PATIENCE = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "train-models", "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "train-models", "Models", "Cleanser")
TEST_FILE = os.path.join(DATA_DIR, "test_cleanser.csv")
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "best_model_bigru_cleanser_focal_loss_VI.pt")

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
POLARITIES = ['positive', 'negative', 'neutral', 'None']
NUM_ASPECTS = len(ASPECTS)
NUM_POLARITIES = len(POLARITIES)
MODEL_NAME = "vinai/phobert-base"


# ==========================================
# 2. XỬ LÝ DỮ LIỆU (DATASET & DATALOADER)
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

        target_matrix = torch.full((NUM_ASPECTS,), 3, dtype=torch.long)

        for label in labels:
            if '#' in label:
                aspect, polarity = label.split('#')
                if aspect in ASPECTS and polarity in POLARITIES:
                    aspect_idx = ASPECTS.index(aspect)
                    polarity_idx = POLARITIES.index(polarity)
                    target_matrix[aspect_idx] = polarity_idx

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': target_matrix
        }


# ==========================================
# 3. KIẾN TRÚC MÔ HÌNH
# ==========================================
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
        logits = logits.view(-1, self.num_aspects, self.num_polarities)
        return logits


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



def parse_labels(raw_label_string):
    if pd.isna(raw_label_string) or not isinstance(raw_label_string, str):
        return []

    parts = raw_label_string.split(';')
    cleaned_labels = [p.strip('{} ') for p in parts if p.strip('{} ')]
    return cleaned_labels


# ==========================================
# 4. HÀM ĐÁNH GIÁ (Đã sửa lỗi trùng lập SPC và ACD+SPC)
# ==========================================
def evaluate_full_absa(model, dataloader, device, desc="Full ABSA Evaluation"):
    model.eval()

    preds_all = []
    targets_all = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=2)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds_all = np.vstack(preds_all)
    targets_all = np.vstack(targets_all)

    preds_flat = preds_all.flatten()
    targets_flat = targets_all.flatten()

    # =========================
    # 1. BƯỚC CHUẨN BỊ MA TRẬN 33 LỚP (Cho tính toán chi tiết)
    # =========================
    num_samples = targets_all.shape[0]
    y_true_bin = np.zeros((num_samples, NUM_ASPECTS * 3), dtype=int)
    y_pred_bin = np.zeros((num_samples, NUM_ASPECTS * 3), dtype=int)
    target_names = []

    for i, aspect in enumerate(ASPECTS):
        target_names.extend([f"{aspect}#positive", f"{aspect}#negative", f"{aspect}#neutral"])

    for i in range(num_samples):
        for j in range(NUM_ASPECTS):
            t = targets_all[i, j]
            p = preds_all[i, j]
            if t != 3:
                y_true_bin[i, j * 3 + t] = 1
            if p != 3:
                y_pred_bin[i, j * 3 + p] = 1

    # =========================
    # 2. TÍNH TOÁN CÁC CHỈ SỐ THEO TỪNG KHÍA CẠNH
    # =========================
    acd_p_list, acd_r_list, acd_f1_list = [], [], []
    spc_p_list, spc_r_list, spc_f1_list = [], [], []
    support_list = []

    table5_rows = []
    detailed_sentiment_rows = []  # Lưu trước các hàng cho bảng chi tiết cảm xúc

    for i, aspect in enumerate(ASPECTS):
        # --- Aspect Detection (ACD) ---
        t_i = (targets_all[:, i] != 3).astype(int)
        p_i = (preds_all[:, i] != 3).astype(int)

        support = int(np.sum(t_i))
        support_list.append(support)

        p_acd, r_acd, f1_acd, _ = precision_recall_fscore_support(t_i, p_i, average='macro', zero_division=0)

        acd_p_list.append(p_acd)
        acd_r_list.append(r_acd)
        acd_f1_list.append(f1_acd)

        # --- Sentiment Detection (Đánh giá kết hợp khắt khe ACD+SPC cho Bảng 5) ---
        idx_pos = i * 3 + 0
        idx_neg = i * 3 + 1
        idx_neu = i * 3 + 2

        def get_binary_metrics(idx):
            return precision_recall_fscore_support(y_true_bin[:, idx], y_pred_bin[:, idx], average='binary',
                                                   zero_division=0)

        p_pos, r_pos, f1_pos, _ = get_binary_metrics(idx_pos)
        p_neg, r_neg, f1_neg, _ = get_binary_metrics(idx_neg)
        p_neu, r_neu, f1_neu, _ = get_binary_metrics(idx_neu)

        supp_pos = int(np.sum(y_true_bin[:, idx_pos]))
        supp_neg = int(np.sum(y_true_bin[:, idx_neg]))
        supp_neu = int(np.sum(y_true_bin[:, idx_neu]))

        detailed_sentiment_rows.append(
            f"| {aspect + '#positive':<20} | {p_pos * 100:>10.2f}| {r_pos * 100:>9.2f}| {f1_pos * 100:>10.2f}| {supp_pos:>7} |")
        detailed_sentiment_rows.append(
            f"| {aspect + '#negative':<20} | {p_neg * 100:>10.2f}| {r_neg * 100:>9.2f}| {f1_neg * 100:>10.2f}| {supp_neg:>7} |")
        detailed_sentiment_rows.append(
            f"| {aspect + '#neutral':<20} | {p_neu * 100:>10.2f}| {r_neu * 100:>9.2f}| {f1_neu * 100:>10.2f}| {supp_neu:>7} |")
        detailed_sentiment_rows.append("-" * 75)

        # Trung bình cộng 3 nhãn cảm xúc cho Bảng 5
        p_spc = (p_pos + p_neg + p_neu) / 3
        r_spc = (r_pos + r_neg + r_neu) / 3
        f1_spc = (f1_pos + f1_neg + f1_neu) / 3

        spc_p_list.append(p_spc)
        spc_r_list.append(r_spc)
        spc_f1_list.append(f1_spc)

        table5_rows.append(
            f"| {aspect:<15} | {p_acd * 100:>10.2f}| {r_acd * 100:>9.2f}| {f1_acd * 100:>10.2f}| {p_spc * 100:>10.2f}| {r_spc * 100:>9.2f}| {f1_spc * 100:>10.2f}| {support:>7} |")

    # Trung bình ACD
    p_acd_mac, r_acd_mac, f1_acd_mac = np.mean(acd_p_list), np.mean(acd_r_list), np.mean(acd_f1_list)

    # Cột "Sentiment Detection" Bảng 5 bản chất là ACD+SPC, ta lưu vào biến p_pair_mac
    p_pair_mac, r_pair_mac, f1_pair_mac = np.mean(spc_p_list), np.mean(spc_r_list), np.mean(spc_f1_list)
    total_support = sum(support_list)

    # =========================
    # 3. TÍNH TOÁN CÁC CHỈ SỐ MICRO, MACRO, WEIGHTED VÀ ACCURACY CHO BẢNG 1
    # =========================

    # --- ACD (Khía cạnh thuần) ---
    pred_aspect = (preds_flat != 3).astype(int)
    target_aspect = (targets_flat != 3).astype(int)
    p_acd_mic, r_acd_mic, f1_acd_mic, _ = precision_recall_fscore_support(target_aspect, pred_aspect, average="micro",
                                                                          zero_division=0)
    p_acd_wei, r_acd_wei, f1_acd_wei, _ = precision_recall_fscore_support(target_aspect, pred_aspect,
                                                                          average="weighted",
                                                                          zero_division=0)  # Thêm Weighted
    mask_acd = ~((pred_aspect == 0) & (target_aspect == 0))
    acc_acd = accuracy_score(target_aspect[mask_acd], pred_aspect[mask_acd])

    # --- SPC (Cảm xúc thuần) ---
    mask_spc = targets_flat != 3
    preds_spc = preds_flat[mask_spc]
    targets_spc = targets_flat[mask_spc]

    # Tính Macro, Micro, và Weighted thuần túy của 3 nhãn Pos, Neg, Neu
    p_spc_mac, r_spc_mac, f1_spc_mac, _ = precision_recall_fscore_support(targets_spc, preds_spc, labels=[0, 1, 2],
                                                                          average="macro", zero_division=0)
    p_spc_mic, r_spc_mic, f1_spc_mic, _ = precision_recall_fscore_support(targets_spc, preds_spc, labels=[0, 1, 2],
                                                                          average="micro", zero_division=0)
    p_spc_wei, r_spc_wei, f1_spc_wei, _ = precision_recall_fscore_support(targets_spc, preds_spc, labels=[0, 1, 2],
                                                                          average="weighted",
                                                                          zero_division=0)  # Thêm Weighted
    acc_spc = accuracy_score(targets_spc, preds_spc)

    # --- ACD + SPC (Kết hợp 33 classes) ---
    p_pair_mic, r_pair_mic, f1_pair_mic, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="micro",
                                                                             zero_division=0)
    p_pair_wei, r_pair_wei, f1_pair_wei, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="weighted",
                                                                             zero_division=0)  # Thêm Weighted

    # Ghi chú: p_pair_mac đã được gán bằng giá trị trung bình từ Bảng 5 ở trên

    mask_pair = (targets_flat != 3) | (preds_flat != 3)
    acc_pair = accuracy_score(targets_flat[mask_pair], preds_flat[mask_pair])

    # =========================
    # 4. IN KẾT QUẢ TỔNG HỢP (Bảng 1)
    # =========================
    # Mở rộng chiều ngang bảng lên 128 ký tự để chứa thêm cột
    print("\n" + "=" * 128)
    print(f"{'ABSA EVALUATION RESULTS':^128}")
    print("=" * 128)
    header = f"| {'Task':<10} | {'P(Mic)':<8} | {'P(Mac)':<8} | {'P(Wei)':<8} | {'R(Mic)':<8} | {'R(Mac)':<8} | {'R(Wei)':<8} | {'F1(Mic)':<8} | {'F1(Mac)':<8} | {'F1(Wei)':<8} | {'Accuracy':<8} |"
    print(header)
    print("-" * 128)
    print(
        f"| {'ACD':<10} | {p_acd_mic:<8.4f} | {p_acd_mac:<8.4f} | {p_acd_wei:<8.4f} | {r_acd_mic:<8.4f} | {r_acd_mac:<8.4f} | {r_acd_wei:<8.4f} | {f1_acd_mic:<8.4f} | {f1_acd_mac:<8.4f} | {f1_acd_wei:<8.4f} | {acc_acd:<8.4f} |")
    print(
        f"| {'SPC':<10} | {p_spc_mic:<8.4f} | {p_spc_mac:<8.4f} | {p_spc_wei:<8.4f} | {r_spc_mic:<8.4f} | {r_spc_mac:<8.4f} | {r_spc_wei:<8.4f} | {f1_spc_mic:<8.4f} | {f1_spc_mac:<8.4f} | {f1_spc_wei:<8.4f} | {acc_spc:<8.4f} |")
    print(
        f"| {'ACD+SPC':<10} | {p_pair_mic:<8.4f} | {p_pair_mac:<8.4f} | {p_pair_wei:<8.4f} | {r_pair_mic:<8.4f} | {r_pair_mac:<8.4f} | {r_pair_wei:<8.4f} | {f1_pair_mic:<8.4f} | {f1_pair_mac:<8.4f} | {f1_pair_wei:<8.4f} | {acc_pair:<8.4f} |")
    print("=" * 128)

    # =========================
    # 5. IN BẢNG CHUẨN THEO BÀI NGHIÊN CỨU (Table 5)
    # =========================
    print("\n" + "=" * 106)
    print(f"{'Table 5: Performances (%) of our approach in terms of different aspects.':^106}")
    print("=" * 106)
    header1 = f"| {'Aspect':<15} | {'Aspect Detection':^33} | {'Sentiment Detect (ACD+SPC)':^33} | {'Support':^7} |"
    header2 = f"| {'':<15} | {'Precision':<10}| {'Recall':<9}| {'F1-score':<10}| {'Precision':<10}| {'Recall':<9}| {'F1-score':<10}| {'':^7} |"
    print(header1)
    print(header2)
    print("-" * 106)

    for row in table5_rows:
        print(row)

    print("-" * 106)
    print(
        f"| {'Macro Avg':<15} | {p_acd_mac * 100:>10.2f}| {r_acd_mac * 100:>9.2f}| {f1_acd_mac * 100:>10.2f}| {p_pair_mac * 100:>10.2f}| {r_pair_mac * 100:>9.2f}| {f1_pair_mac * 100:>10.2f}| {total_support:>7} |")
    print("=" * 106)

    # =========================
    # 6. IN BẢNG CHI TIẾT CẢM XÚC (Theo từng khía cạnh)
    # =========================
    print("\n" + "=" * 75)
    print(f"{'DETAILED SENTIMENT METRICS PER ASPECT (%)':^75}")
    print("=" * 75)
    print(f"| {'Aspect#Polarity':<20} | {'Precision':<10}| {'Recall':<9}| {'F1-score':<10}| {'Support':^7} |")
    print("-" * 75)

    for row in detailed_sentiment_rows:
        print(row)

    print("=" * 75)


# ==========================================
# 5. THỰC THI (MAIN EXECUTION)
# ==========================================
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    df_filtered = pd.read_csv(TEST_FILE, encoding="utf-8")

    # Bóc tách nhãn
    df_filtered['parsed_labels'] = df_filtered['label'].apply(parse_labels)

    df_test = df_filtered

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model2 = PhoBERT_BiGRU(num_aspects=NUM_ASPECTS, num_polarities=NUM_POLARITIES)
    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model2.load_state_dict(checkpoint)
    model2.to(device)

    # 1. ĐÁNH GIÁ TỔNG TOÀN BỘ TẬP TEST (Nếu bạn vẫn muốn xem kết quả chung)
    print("\n" + "*" * 100)
    print(f"{'ĐÁNH GIÁ TRÊN TOÀN BỘ TẬP TEST (ALL PLATFORMS)':^100}")
    print("*" * 100)
    test_dataset_all = ABSADataset(df_test['cleaned_comment'].tolist(), df_test['parsed_labels'].tolist(), tokenizer)
    test_loader_all = DataLoader(test_dataset_all, batch_size=BATCH_SIZE, shuffle=False)
    evaluate_full_absa(model2, test_loader_all, device, desc="Evaluating All Platforms")

