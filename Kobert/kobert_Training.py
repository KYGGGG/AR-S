import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from torch import nn
from torch.optim import AdamW
from tqdm.auto import tqdm
from kobert_tokenizer import KoBERTTokenizer
import joblib
import json
from pathlib import Path
import os

# Local training utilities
from kobert_data_utils import KoBERTMultiTaskDataset, collate_fn, KoBERTMultiTaskClassifier, preprocess_and_encode_data
from Kobert_Train_Val import train_epoch, eval_model
from utils import EarlyStopping, set_seed

PROJECT_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
DATA_DIR = PROJECT_DIR / "Kobert" / "Data"

df = pd.read_csv(DATA_DIR / 'merged_all.csv')
df = df[['processed_consulting_content_combined_for_kobert', 'source', 'consulting_category']].dropna()
set_seed(1234)
# Encode labels and build the hierarchical source-to-category mapping.
df, source_encoder, category_encoder, main_to_sub_category_map = preprocess_and_encode_data(df)

train_df, val_df = train_test_split(df, test_size=0.2, random_state=1234)
num_sources = len(source_encoder.classes_)
num_categories = len(category_encoder.classes_)

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

train_dataset = KoBERTMultiTaskDataset(
    texts=train_df['processed_consulting_content_combined_for_kobert'].to_numpy(),
    sources=train_df['source_encoded'].to_numpy(),
    categories=train_df['category_encoded'].to_numpy(),
    tokenizer=tokenizer
)

val_dataset = KoBERTMultiTaskDataset(
    texts=val_df['processed_consulting_content_combined_for_kobert'].to_numpy(),
    sources=val_df['source_encoded'].to_numpy(),
    categories=val_df['category_encoded'].to_numpy(),
    tokenizer=tokenizer
)
BATCH_SIZE = 32

# Bind the tokenizer when constructing each DataLoader.
train_data_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda batch: collate_fn(batch, tokenizer)
)

val_data_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    collate_fn=lambda batch: collate_fn(batch, tokenizer)
)

# Initialize the model, loss functions, and optimizer.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KoBERTMultiTaskClassifier(num_sources, num_categories).to(device)

optimizer = AdamW(model.parameters(), lr=4e-5)
source_criterion = nn.CrossEntropyLoss()
category_criterion = nn.CrossEntropyLoss()

EPOCHS = 50

early_stopping = EarlyStopping(
    patience=5,
    verbose=True,
    delta=0.1,
    path=str(DATA_DIR / 'checkpoint.pt'),
)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    train_loss, train_source_acc, train_category_acc = train_epoch(
        model,
        train_data_loader,
        optimizer,
        source_criterion,
        category_criterion,
        device,
        main_to_sub_category_map
    )
    print(f'Train Loss: {train_loss:.4f}, Train Source Acc: {train_source_acc:.4f}, Train Category Acc: {train_category_acc:.4f}')

    val_loss, val_source_acc, val_category_acc, \
    all_source_preds, all_source_labels, \
    all_category_preds, all_category_labels, \
    all_texts = eval_model( 
        model,
        val_data_loader,
        source_criterion,
        category_criterion,
        device,
        main_to_sub_category_map
    )
    print(f'Val Loss: {val_loss:.4f}, Val Source Acc: {val_source_acc:.4f}, Val Category Acc: {val_category_acc:.4f}')

    early_stopping(val_loss, model)
    
    if early_stopping.early_stop:
        print("Early stopping")
        break
