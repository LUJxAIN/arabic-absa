# import libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

# ---- Step 1: Configuration Setup ----
MODEL_NAME = "UBC-NLP/MARBERTv2"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
TEMPERATURE = 1.5      # softens Transformer overconfidence boundaries
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASPECT_COLUMNS = ['price', 'cleanliness', 'facilities', 'service', 'environment', 'overall']
NUM_CLASSES = 2  # binary classes (negative, positive)
LABEL_MAP = {-1: 0, 1: 1}


# ---- Step 2: Data preprocessing functions ----
def clean_arabic_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)
    text = re.sub(r'[\u064B-\u0652]', '', text)
    text = re.sub(r'([!@#$%^&*(),.?":{}|<>])', r' \1 ', text)
    return re.sub(r'\s+', ' ', text).strip()

def safe_int_parse(val):
    if pd.isna(val) or str(val).strip().lower() in ['nan', 'null', '']: return None
    s = str(val).strip()
    if s.endswith('-'): s = '-' + s[:-1].strip()
    try: return int(float(s))
    except ValueError: return None

def create_stratification_key(df, aspect_cols):
    """Generates a binary-aware multi-target composite key for stratification."""
    temp_df = df[aspect_cols].copy()
    for col in aspect_cols:
        val_parsed = temp_df[col].apply(safe_int_parse)
        # groups missing data and neutral instances together since binary drops both
        temp_df[col] = val_parsed.apply(lambda x: str(x) if x in [-1, 1] else 'X')
    composite = temp_df[aspect_cols].agg('_'.join, axis=1)
    counts = composite.value_counts()
    return composite.replace(counts[counts < 2].index, 'Rare_Group')

def manual_text_oversample(corpus, targets):
    df_temp = pd.DataFrame({'text': corpus, 'target': targets})
    class_counts = df_temp['target'].value_counts()
    if len(class_counts) <= 1: return df_temp['text'].values, df_temp['target'].values
    max_count = class_counts.max()
    resampled_rows = []
    for class_label, count in class_counts.items():
        sub_df = df_temp[df_temp['target'] == class_label]
        rep = max_count // count
        rem = max_count % count
        resampled_rows.append(sub_df.loc[sub_df.index.repeat(rep)])
        if rem > 0: resampled_rows.append(sub_df.sample(n=rem, replace=True, random_state=42))
    final_df = pd.concat(resampled_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return final_df['text'].values, final_df['target'].values


# ---- Step 3: Dataset and model structure ----
class ArabicBinaryDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        text_cleaned = clean_arabic_text(str(row['processed_reviews']))
        inputs = self.tokenizer(text_cleaned, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')

        labels = []
        for col in ASPECT_COLUMNS:
            val = safe_int_parse(row[col])
            if val is None or val not in LABEL_MAP:
                labels.append(-100)
            else:
                labels.append(LABEL_MAP[val])

        return {'input_ids': inputs['input_ids'].flatten(), 'attention_mask': inputs['attention_mask'].flatten(), 'labels': torch.tensor(labels, dtype=torch.long)}

class MultiHeadBinaryMARBERTv2(nn.Module):
    def __init__(self, model_name, num_aspects, num_classes):
        super(MultiHeadBinaryMARBERTv2, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Configured strictly for 2 output classes
        self.heads = nn.ModuleList([nn.Linear(self.bert.config.hidden_size, num_classes) for _ in range(num_aspects)])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        return [head(pooled_output) for head in self.heads]


# ---- Step 4: Model Training ----
def train_marbert(model, data_loader, optimizer, scheduler, criteria, device):
    model.train()
    for batch in data_loader:
        input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=mask)
        loss = sum(criterion(outputs[i], labels[:, i]) for i, criterion in enumerate(criteria)) / len(ASPECT_COLUMNS)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

def extract_calibrated_probs(model, data_loader, device, temp_scalar=1.0):
    model.eval()
    prob_dict = {i: [] for i in range(len(ASPECT_COLUMNS))}
    label_dict = {i: [] for i in range(len(ASPECT_COLUMNS))}
    with torch.no_grad():
        for batch in data_loader:
            input_ids, mask, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].numpy()
            outputs = model(input_ids=input_ids, attention_mask=mask)
            for i in range(len(ASPECT_COLUMNS)):
                scaled_logits = outputs[i] / temp_scalar
                probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()
                prob_dict[i].extend(probs)
                label_dict[i].extend(labels[:, i])
    return {i: np.array(prob_dict[i]) for i in range(len(ASPECT_COLUMNS))}, {i: np.array(label_dict[i]) for i in range(len(ASPECT_COLUMNS))}


# ---- Step 5: Final system pipeline ----
if __name__ == "__main__":
    df = pd.read_excel("golden_data_set_processed_google_reviews-2.xlsx")
    df.rename(columns={"السعر_golden": "price", "النظافة_golden": "cleanliness",
                          "المرافق_golden": "facilities", "الخدمة و الموظفين_golden": "service", "البيئة_golden": "environment",
                          "التجربة بشكل عام_golden": "overall"}, inplace=True)
    for col in ASPECT_COLUMNS: df[col] = df[col].apply(safe_int_parse)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    strat_vector = create_stratification_key(df, ASPECT_COLUMNS)

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=strat_vector)
    sub_train_df, blend_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=create_stratification_key(train_df, ASPECT_COLUMNS))

    # # ---- 1. Binary MARBERTv2 ----
    print(">>> Training Binary MARBERTv2 Context Layer...")
    sub_train_loader = DataLoader(ArabicBinaryDataset(sub_train_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE, shuffle=True)
    blend_loader = DataLoader(ArabicBinaryDataset(blend_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)
    test_loader = DataLoader(ArabicBinaryDataset(test_df, tokenizer, MAX_LEN), batch_size=BATCH_SIZE)

    marbert_model = MultiHeadBinaryMARBERTv2(MODEL_NAME, num_aspects=len(ASPECT_COLUMNS), num_classes=NUM_CLASSES).to(DEVICE)

    weights = []
    for col in ASPECT_COLUMNS:
        v_labels = sub_train_df[col].dropna()
        v_labels = v_labels[v_labels.isin(LABEL_MAP.keys())].map(LABEL_MAP).astype(int).values
        w = compute_class_weight(class_weight='balanced', classes=np.array([0,1]), y=v_labels) if len(v_labels) > 0 else np.ones(NUM_CLASSES)
        weights.append(torch.tensor(w, dtype=torch.float).to(DEVICE))

    criteria = [nn.CrossEntropyLoss(weight=w, ignore_index=-100) for w in weights]

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_params = [
        {'params': [p for n, p in marbert_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in marbert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_params, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * (len(sub_train_loader)*EPOCHS)), num_training_steps=len(sub_train_loader)*EPOCHS)

    for epoch in range(EPOCHS):
        train_marbert(marbert_model, sub_train_loader, optimizer, scheduler, criteria, DEVICE)

    mar_blend_probs, blend_labels = extract_calibrated_probs(marbert_model, blend_loader, DEVICE, temp_scalar=TEMPERATURE)
    mar_test_probs, test_labels = extract_calibrated_probs(marbert_model, test_loader, DEVICE, temp_scalar=TEMPERATURE)

    # # ---- 2. Binary LinearSVC ----
    print(">>> Training Binary LinearSVC Keyword Layer...")
    sub_train_corpus = sub_train_df['processed_reviews'].astype(str).apply(clean_arabic_text).values
    blend_corpus = blend_df['processed_reviews'].astype(str).apply(clean_arabic_text).values
    test_corpus = test_df['processed_reviews'].astype(str).apply(clean_arabic_text).values

    svc_results_test = {}
    svc_blend_scores = {}
    svc_test_scores = {}

    for i, col in enumerate(ASPECT_COLUMNS):
        y_sub_tr_raw = sub_train_df[col].values

        # Binary filter: Mask out rows containing NaNs OR Neutral (0) values
        tr_mask = (~pd.isna(y_sub_tr_raw)) & (np.isin(y_sub_tr_raw, [-1, 1]))

        X_tr_text = sub_train_corpus[tr_mask]
        y_tr_mapped = np.array([LABEL_MAP[v] for v in y_sub_tr_raw[tr_mask]])

        X_tr_man, y_tr_man = manual_text_oversample(X_tr_text, y_tr_mapped)

        tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
        X_tr_tfidf = tfidf.fit_transform(X_tr_man)
        X_blend_tfidf = tfidf.transform(blend_corpus)
        X_test_tfidf = tfidf.transform(test_corpus)

        clf = LinearSVC(C=1.0, dual=False, max_iter=10000, random_state=42)
        if len(y_tr_man) > 0:
            clf.fit(X_tr_tfidf, y_tr_man)
            svc_blend_scores[i] = clf.decision_function(X_blend_tfidf)
            svc_test_scores[i] = clf.decision_function(X_test_tfidf)
            svc_results_test[i] = clf.predict(X_test_tfidf)
        else:
            svc_blend_scores[i] = np.zeros(len(blend_corpus))
            svc_test_scores[i] = np.zeros(len(test_corpus))
            svc_results_test[i] = np.zeros(len(test_corpus))

    # --- 3. Binary Meta-Learner Fusion Stacking ---
    print(">>> Running Binary Meta-Learner Fusion...")
    ensemble_final_predictions = {}
    cleaned_test_targets = {}

    for i, col in enumerate(ASPECT_COLUMNS):
        bl_mask = blend_labels[i] != -100
        ts_mask = test_labels[i] != -100

        # multi-dimensional scaling alignment on single SVM output rows
        s_blend_feat = svc_blend_scores[i].reshape(-1, 1) if svc_blend_scores[i].ndim == 1 else svc_blend_scores[i]
        s_test_feat = svc_test_scores[i].reshape(-1, 1) if svc_test_scores[i].ndim == 1 else svc_test_scores[i]

        X_blend_meta = np.hstack([mar_blend_probs[i], s_blend_feat])
        X_test_meta = np.hstack([mar_test_probs[i], s_test_feat])

        y_blend_meta = blend_labels[i][bl_mask]
        y_test_meta = test_labels[i][ts_mask]
        cleaned_test_targets[i] = y_test_meta

        meta_learner = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

        if len(y_blend_meta) > 0 and len(np.unique(y_blend_meta)) > 1:
            meta_learner.fit(X_blend_meta[bl_mask], y_blend_meta)
            ensemble_final_predictions[i] = meta_learner.predict(X_test_meta[ts_mask])
        else:
            ensemble_final_predictions[i] = np.argmax(mar_test_probs[i][ts_mask], axis=1)

    # metric display
    print("\n" + "="*85)
    print(f"{'BINARY ASPECT':<15} | {'MODEL':<15} | {'ACCURACY':<10} | {'MACRO PREC':<12} | {'MACRO REC':<12} | {'MACRO F1':<10}")
    print("="*85)

    for i, col in enumerate(ASPECT_COLUMNS):
        t_labels = cleaned_test_targets[i]
        if len(t_labels) == 0: continue
        labels_present = np.unique(t_labels)
        ts_mask = test_labels[i] != -100

        # MARBERTv2 results
        m_preds = np.argmax(mar_test_probs[i], axis=1)[ts_mask]
        m_acc = accuracy_score(t_labels, m_preds)
        mp, mr, mf, _ = precision_recall_fscore_support(t_labels, m_preds, average='macro', labels=labels_present, zero_division=0)
        print(f"{col.upper():<15} | {'MARBERTv2 Only':<15} | {m_acc:<10.4f} | {mp:<12.4f} | {mr:<12.4f} | {mf:<10.4f}")

        # LinearSVC results
        s_preds = svc_results_test[i][ts_mask]
        s_acc = accuracy_score(t_labels, s_preds)
        sp, sr, sf, _ = precision_recall_fscore_support(t_labels, s_preds, average='macro', labels=labels_present, zero_division=0)
        print(f"{'':<15} | {'LinearSVC Only':<15} | {s_acc:<10.4f} | {sp:<12.4f} | {sr:<12.4f} | {sf:<10.4f}")

        # Decision Stacking Ensemble results
        e_preds = ensemble_final_predictions[i]
        e_acc = accuracy_score(t_labels, e_preds)
        ep, er, ef, _ = precision_recall_fscore_support(t_labels, e_preds, average='macro', labels=labels_present, zero_division=0)
        print(f"{'':<15} | {'BINARY STACKING':<15} | {e_acc:<10.4f} | {ep:<12.4f} | {er:<12.4f} | {ef:<10.4f}")
        print("-" * 85)
