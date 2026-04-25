# cap_dev_caption_inception_fixed.py - Fixed torch.load issue

import os
import sys
import argparse
import logging
import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm
from collections import Counter
import re
import random
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_dev_caption.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# For evaluation
from bert_score import score as bert_score
from rouge_score import rouge_scorer

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class Vocabulary:
    """Vocabulary class for caption generation"""
    
    def __init__(self, freq_threshold=3):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        
    def __len__(self):
        return len(self.itos)
    
    def tokenize(self, text):
        """Simple tokenization"""
        text = str(text).lower()
        text = re.sub(r'[^\w\s\-\.\(\)\/]', ' ', text)
        tokens = text.split()
        return tokens
    
    def build_vocabulary(self, captions):
        """Build vocabulary from list of captions"""
        frequencies = Counter()
        
        for caption in captions:
            if caption and not pd.isna(caption):
                tokens = self.tokenize(caption)
                frequencies.update(tokens)
        
        logger.info(f"Total unique words: {len(frequencies)}")
        
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold and word not in self.stoi:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
                
        logger.info(f"Vocabulary size: {len(self.itos)}")
    
    def numericalize(self, text, max_length=80):
        """Convert text to numerical tokens"""
        tokens = self.tokenize(text)
        tokens = ["<SOS>"] + tokens[:max_length-2] + ["<EOS>"]
        indices = [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
        return indices


class DevCaptionDataset(Dataset):
    """Dataset for dev_caption with train/val/test splits"""
    
    def __init__(self, data_root, split='train', vocab=None, transform=None, 
                 max_length=80, max_samples=None):
        self.data_root = data_root
        self.split = split
        self.vocab = vocab
        self.transform = transform or self._get_default_transform()
        self.max_length = max_length
        
        # Load captions
        captions_file = os.path.join(data_root, 'captions.csv')
        self.df = pd.read_csv(captions_file)
        
        # Separate train and valid images
        self.train_df = self.df[self.df['ID'].str.contains('train', na=False)]
        self.valid_df = self.df[self.df['ID'].str.contains('valid', na=False)]
        
        logger.info(f"Total train images: {len(self.train_df)}")
        logger.info(f"Total valid images: {len(self.valid_df)}")
        
        # Split train into train (80%) and val (20%)
        train_ids = self.train_df['ID'].tolist()
        train_captions = self.train_df['Caption'].tolist()
        
        train_ids_split, val_ids_split, train_caps_split, val_caps_split = train_test_split(
            train_ids, train_captions, test_size=0.2, random_state=42
        )
        
        # Create dataframes for each split
        self.train_split_df = pd.DataFrame({'ID': train_ids_split, 'Caption': train_caps_split})
        self.val_split_df = pd.DataFrame({'ID': val_ids_split, 'Caption': val_caps_split})
        self.test_df = self.valid_df  # Actual valid images are test set
        
        logger.info(f"New train (80% of original train): {len(self.train_split_df)}")
        logger.info(f"New val (20% of original train): {len(self.val_split_df)}")
        logger.info(f"Test (original valid): {len(self.test_df)}")
        
        # Select data based on split
        if split == 'train':
            self.data = self.train_split_df
        elif split == 'val':
            self.data = self.val_split_df
        else:  # test
            self.data = self.test_df
        
        # Limit samples if specified
        if max_samples and max_samples < len(self.data):
            self.data = self.data.sample(n=max_samples, random_state=42)
        
        logger.info(f"Split '{split}' using {len(self.data)} samples")
        
    def _get_default_transform(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_id = row['ID']
        caption = row['Caption']
        
        # Load image
        img_path = os.path.join(self.data_root, 'images', f"{img_id}.jpg")
        
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            image_tensor = torch.randn(3, 299, 299)
        
        # Process caption
        if self.vocab and caption and not pd.isna(caption):
            tokens = self.vocab.numericalize(caption, self.max_length)
            caption_tensor = torch.tensor(tokens, dtype=torch.long)
        else:
            caption_tensor = torch.tensor([0], dtype=torch.long)
        
        return {
            'image': image_tensor,
            'caption': caption_tensor,
            'caption_text': str(caption) if caption and not pd.isna(caption) else "",
            'id': img_id
        }


def collate_fn(batch):
    """Custom collate function for variable length captions"""
    images = torch.stack([item['image'] for item in batch])
    captions = [item['caption'] for item in batch]
    caption_texts = [item['caption_text'] for item in batch]
    ids = [item['id'] for item in batch]
    
    # Pad captions to max length in batch
    max_len = max(len(cap) for cap in captions)
    padded_captions = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded_captions[i, :len(cap)] = cap
    
    return {
        'image': images,
        'caption': padded_captions,
        'caption_text': caption_texts,
        'id': ids
    }


class InceptionV3Encoder(nn.Module):
    """InceptionV3 Encoder"""
    
    def __init__(self, embed_size=512, pretrained=True):
        super(InceptionV3Encoder, self).__init__()
        
        logger.info("Loading InceptionV3 encoder...")
        
        # Try to load pretrained InceptionV3
        try:
            cache_path = os.path.expanduser('~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth')
            if os.path.exists(cache_path):
                logger.info("Found cached InceptionV3")
                self.inception = models.inception_v3(weights=None, aux_logits=False)
                state_dict = torch.load(cache_path, map_location='cpu', weights_only=True)
                state_dict = {k: v for k, v in state_dict.items() if 'AuxLogits' not in k}
                self.inception.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pretrained InceptionV3 from cache")
            else:
                logger.warning("No cached InceptionV3, using random initialization")
                self.inception = models.inception_v3(weights=None, aux_logits=False)
        except Exception as e:
            logger.warning(f"Failed to load pretrained InceptionV3: {e}")
            self.inception = models.inception_v3(weights=None, aux_logits=False)
        
        # Remove final layer
        self.inception.fc = nn.Identity()
        
        # Projection layer
        self.fc = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(0.3)
        
        # Freeze early layers, train only later layers
        for name, param in self.inception.named_parameters():
            if 'Mixed_6' not in name and 'Mixed_7' not in name:
                param.requires_grad = False
                
        logger.info(f"InceptionV3 initialized. Embed size: {embed_size}")
        
    def forward(self, images):
        features = self.inception(images)
        if isinstance(features, tuple):
            features = features[0]
        features = self.fc(features)
        features = self.bn(features)
        features = self.dropout(features)
        return features


class CaptionDecoder(nn.Module):
    """LSTM Decoder for caption generation"""
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2):
        super(CaptionDecoder, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, features, captions):
        """Forward pass for training"""
        batch_size = features.size(0)
        device = features.device
        
        # Initialize hidden state with image features
        h0 = features.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)
        
        # Embed captions (exclude last token)
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        
        # LSTM forward
        outputs, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(self.dropout(outputs))
        
        return outputs
    
    def generate(self, features, max_length=80, vocab=None):
        """Generate captions for inference"""
        batch_size = features.size(0)
        device = features.device
        
        # Start with SOS token
        inputs = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
        
        hidden = (features.unsqueeze(0).repeat(self.num_layers, 1, 1),
                  torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))
        
        generated = []
        
        for _ in range(max_length):
            embeddings = self.embed(inputs)
            outputs, hidden = self.lstm(embeddings, hidden)
            outputs = self.fc(outputs)
            
            _, predicted = outputs.max(2)
            generated.append(predicted)
            inputs = predicted
            
        generated = torch.cat(generated, dim=1)
        return generated


def decode_caption(tokens, vocab):
    """Convert token indices to text"""
    words = []
    for token in tokens:
        if token == 2:  # <EOS>
            break
        if token > 3:  # Skip special tokens
            word = vocab.itos.get(token, '<UNK>')
            words.append(word)
    return ' '.join(words) if words else "Radiology image"


class Trainer:
    """Trainer for InceptionV3 + LSTM"""
    
    def __init__(self, encoder, decoder, train_loader, val_loader, vocab, config):
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab = vocab
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        # Optimizer with different learning rates
        params = [
            {'params': self.encoder.fc.parameters(), 'lr': config['learning_rate']},
            {'params': [p for n, p in self.encoder.inception.named_parameters() 
                       if p.requires_grad], 'lr': config['learning_rate'] * 0.1},
            {'params': self.decoder.parameters(), 'lr': config['learning_rate']}
        ]
        
        self.optimizer = torch.optim.AdamW(params, weight_decay=0.01)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self, epoch):
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            images = batch['image'].to(self.device)
            captions = batch['caption'].to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.amp.autocast('cuda'):
                    features = self.encoder(images)
                    outputs = self.decoder(features, captions)
                    loss = self.criterion(
                        outputs.view(-1, outputs.size(-1)),
                        captions[:, 1:].contiguous().view(-1)
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                features = self.encoder(images)
                outputs = self.decoder(features, captions)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    captions[:, 1:].contiguous().view(-1)
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), 5.0)
                self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                captions = batch['caption'].to(self.device)
                
                features = self.encoder(images)
                outputs = self.decoder(features, captions)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    captions[:, 1:].contiguous().view(-1)
                )
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def train(self):
        logger.info("Starting training...")
        
        for epoch in range(1, self.config['epochs'] + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch}/{self.config['epochs']}")
            logger.info(f"{'='*50}")
            
            train_loss = self.train_epoch(epoch)
            logger.info(f"Training Loss: {train_loss:.4f}")
            
            val_loss = self.validate()
            logger.info(f"Validation Loss: {val_loss:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                os.makedirs(self.config['save_dir'], exist_ok=True)
                torch.save({
                    'encoder': self.encoder.state_dict(),
                    'decoder': self.decoder.state_dict(),
                    'vocab': self.vocab,
                    'epoch': epoch,
                }, os.path.join(self.config['save_dir'], 'best_model_inception.pt'))
                logger.info(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config['patience']:
                logger.info(f"Early stopping after {epoch} epochs")
                break
                
        logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/scratch/pdnaga.venkatsohana.cse23.itbhu/CAPTION/dev_caption')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--save_dir', type=str, default='/scratch/pdnaga.venkatsohana.cse23.itbhu/CAPTION/checkpoints')
    parser.add_argument('--output_dir', type=str, default='/scratch/pdnaga.venkatsohana.cse23.itbhu/CAPTION/results')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--embed_size', type=int, default=512)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--max_train_samples', type=int, default=40000)
    parser.add_argument('--max_val_samples', type=int, default=10000)
    parser.add_argument('--max_test_samples', type=int, default=None)

    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("="*60)
    logger.info("ImageCLEFmedical Caption 2026 - InceptionV3 Encoder")
    logger.info("="*60)

    # Load training data to build vocabulary
    logger.info("\nLoading training data for vocabulary...")
    captions_file = os.path.join(args.data_root, 'captions.csv')
    df = pd.read_csv(captions_file)
    train_df = df[df['ID'].str.contains('train', na=False)]
    
    # Build vocabulary
    vocab = Vocabulary(freq_threshold=3)
    train_captions = train_df['Caption'].dropna().tolist()
    vocab.build_vocabulary(train_captions)
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = DevCaptionDataset(
        args.data_root, split='train', vocab=vocab, 
        max_length=args.max_length, max_samples=args.max_train_samples
    )
    val_dataset = DevCaptionDataset(
        args.data_root, split='val', vocab=vocab,
        max_length=args.max_length, max_samples=args.max_val_samples
    )
    test_dataset = DevCaptionDataset(
        args.data_root, split='test', vocab=vocab,
        max_length=args.max_length, max_samples=args.max_test_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    logger.info(f"\nFinal dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)}")
    logger.info(f"  Val: {len(val_dataset)}")
    logger.info(f"  Test: {len(test_dataset)}")
    
    # Create models
    logger.info("\nCreating models...")
    encoder = InceptionV3Encoder(embed_size=512)
    decoder = CaptionDecoder(
        embed_size=512,
        hidden_size=512,
        vocab_size=len(vocab),
        num_layers=2
    )
    
    # Train
    config = vars(args)
    trainer = Trainer(encoder, decoder, train_loader, val_loader, vocab, config)
    trainer.train()
    
    # Load best model - FIXED: added weights_only=False
    logger.info("\nLoading best model for evaluation...")
    checkpoint_path = os.path.join(args.save_dir, 'best_model_inception.pt')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    
    # Generate predictions for validation set
    logger.info("\nGenerating predictions for validation set...")
    val_predictions = []
    val_references = []
    val_ids = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            caption_texts = batch['caption_text']
            batch_ids = batch['id']
            
            features = encoder(images)
            generated_tokens = decoder.generate(features, max_length=args.max_length)
            
            for i, tokens in enumerate(generated_tokens):
                caption = decode_caption(tokens.cpu().tolist(), vocab)
                val_predictions.append(caption if caption else "Radiology image")
                val_references.append(caption_texts[i])
                val_ids.append(batch_ids[i])
    
    # Compute metrics on validation set
    logger.info("\nComputing evaluation metrics on validation set...")
    try:
        P, R, F1 = bert_score(val_predictions, val_references, lang='en', verbose=True)
        bertscore_f1 = F1.mean().item()
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge1, rouge2, rougeL = [], [], []
        
        for pred, ref in zip(val_predictions, val_references):
            scores = scorer.score(ref, pred)
            rouge1.append(scores['rouge1'].fmeasure)
            rouge2.append(scores['rouge2'].fmeasure)
            rougeL.append(scores['rougeL'].fmeasure)
        
        metrics = {
            'BERTScore': bertscore_f1,
            'ROUGE-1': np.mean(rouge1),
            'ROUGE-2': np.mean(rouge2),
            'ROUGE-L': np.mean(rougeL)
        }
        
        logger.info("\n" + "="*50)
        logger.info("VALIDATION METRICS")
        logger.info("="*50)
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        metrics = {}
    
    # Generate predictions for test set
    logger.info("\nGenerating predictions for test set...")
    test_predictions = []
    test_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images = batch['image'].to(device)
            batch_ids = batch['id']
            
            features = encoder(images)
            generated_tokens = decoder.generate(features, max_length=args.max_length)
            
            for i, tokens in enumerate(generated_tokens):
                caption = decode_caption(tokens.cpu().tolist(), vocab)
                test_predictions.append(caption if caption else "Radiology image")
                test_ids.append(batch_ids[i])
    
    # Save results
    test_df = pd.DataFrame({
        'ID': test_ids,
        'Generated_Caption': test_predictions
    })
    test_df.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
    
    val_df = pd.DataFrame({
        'ID': val_ids,
        'Generated_Caption': val_predictions,
        'Reference_Caption': val_references
    })
    val_df.to_csv(os.path.join(args.output_dir, 'validation_predictions.csv'), index=False)
    
    if metrics:
        with open(os.path.join(args.output_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {args.output_dir}/")
    logger.info(f"  - test_predictions.csv ({len(test_predictions)} predictions)")
    logger.info(f"  - validation_predictions.csv ({len(val_predictions)} predictions)")
    if metrics:
        logger.info(f"  - evaluation_metrics.json")
    
    logger.info("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
