# mlp_optimized.py
# MLP - Multi-Layer Perceptron for Malware Detection

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import (accuracy_score, f1_score, 
                            confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import joblib
from datetime import datetime
from tqdm import tqdm
from rich.console import Console
from rich.table import Table
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

console = Console()

# ===================== CẤU HÌNH =====================
INPUT_DIR = 'D:/Final_keylogger_ML_2/processed_features_cleaned'
OUTPUT_DIR = 'D:/Final_keylogger_ML_2/MLP'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Train/Test Split
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# MLP ARCHITECTURE
MLP_ARCHITECTURE = {
    'input_neurons': 1864,      # Số features
    'hidden1_neurons': 1024,    # Hidden Layer 1
    'hidden2_neurons': 512,     # Hidden Layer 2
    'hidden3_neurons': 256,     # Hidden Layer 3
    'output_neurons': 1,        # Output Layer (Sigmoid)
    'activation': 'relu',       # Activation cho hidden layers
    'output_activation': 'sigmoid',
    'dropout_rate': 0.3,        # Dropout để tránh overfitting
}

# TRAINING CONFIGURATION
BATCH_SIZE = 256
MAX_EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10
MIN_DELTA = 0.0001

# OPTIMIZER
OPTIMIZER = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

console.print("="*80, style="bold cyan")
console.print("MLP - MULTI-LAYER PERCEPTRON", style="bold cyan", justify="center")
console.print("="*80, style="bold cyan")
console.print(f"[yellow]Dataset:[/yellow] EMBER 2018 (Malware Detection)")
console.print(f"[yellow]Architecture:[/yellow]")
console.print(f"  • Input Layer:    {MLP_ARCHITECTURE['input_neurons']} neurons")
console.print(f"  • Hidden Layer 1: {MLP_ARCHITECTURE['hidden1_neurons']} neurons (ReLU)")
console.print(f"  • Hidden Layer 2: {MLP_ARCHITECTURE['hidden2_neurons']} neurons (ReLU)")
console.print(f"  • Hidden Layer 3: {MLP_ARCHITECTURE['hidden3_neurons']} neurons (ReLU)")
console.print(f"  • Output Layer:   {MLP_ARCHITECTURE['output_neurons']} neuron (Sigmoid)")
console.print(f"[yellow]Training Configuration:[/yellow]")
console.print(f"  • Batch Size: {BATCH_SIZE}")
console.print(f"  • Learning Rate: {LEARNING_RATE}")
console.print(f"  • Max Epochs: {MAX_EPOCHS}")
console.print(f"  • Early Stopping Patience: {PATIENCE}")
console.print("="*80, style="bold cyan")

# ===================== LOAD DATA =====================
def load_full_dataset():
    """Load TOÀN BỘ dataset"""
    console.print("\n[bold yellow]⏳ Loading dataset...[/bold yellow]")
    
    feature_files = [f for f in os.listdir(INPUT_DIR) 
                    if f.endswith('_cleaned.csv') and 'metadata' not in f]
    
    dfs = []
    for file in tqdm(sorted(feature_files), desc="Loading features"):
        filepath = os.path.join(INPUT_DIR, file)
        df = pd.read_csv(filepath, dtype=np.float32)
        dfs.append(df)
    
    X_full = pd.concat(dfs, axis=1)
    
    meta_path = os.path.join(INPUT_DIR, 'metadata_cleaned.csv')
    y_full = pd.read_csv(meta_path, usecols=['label'])['label'].values.astype(np.int8)
    
    memory_gb = X_full.memory_usage(deep=True).sum() / 1024**3
    console.print(f"[green]✓ Loaded:[/green] {X_full.shape[0]:,} samples × {X_full.shape[1]} features")
    console.print(f"[green]✓ Memory:[/green] {memory_gb:.2f} GB")
    console.print(f"[green]✓ Distribution:[/green] Benign={np.sum(y_full==0):,} ({np.sum(y_full==0)/len(y_full)*100:.1f}%), "
                 f"Malware={np.sum(y_full==1):,} ({np.sum(y_full==1)/len(y_full)*100:.1f}%)")
    
    return X_full, y_full

# ===================== BUILD MODEL =====================
def build_mlp_model(input_dim):
    """Xây dựng mô hình MLP"""
    model = models.Sequential([
        # Input Layer
        layers.Input(shape=(input_dim,)),
        
        # Hidden Layer 1: 1024 neurons
        layers.Dense(
            MLP_ARCHITECTURE['hidden1_neurons'], 
            activation=MLP_ARCHITECTURE['activation'],
            kernel_initializer='he_normal',
            name='hidden_layer_1'
        ),
        layers.BatchNormalization(),
        layers.Dropout(MLP_ARCHITECTURE['dropout_rate']),
        
        # Hidden Layer 2: 512 neurons
        layers.Dense(
            MLP_ARCHITECTURE['hidden2_neurons'], 
            activation=MLP_ARCHITECTURE['activation'],
            kernel_initializer='he_normal',
            name='hidden_layer_2'
        ),
        layers.BatchNormalization(),
        layers.Dropout(MLP_ARCHITECTURE['dropout_rate']),
        
        # Hidden Layer 3: 256 neurons
        layers.Dense(
            MLP_ARCHITECTURE['hidden3_neurons'], 
            activation=MLP_ARCHITECTURE['activation'],
            kernel_initializer='he_normal',
            name='hidden_layer_3'
        ),
        layers.BatchNormalization(),
        layers.Dropout(MLP_ARCHITECTURE['dropout_rate']),
        
        # Output Layer: 1 neuron (Sigmoid)
        layers.Dense(
            MLP_ARCHITECTURE['output_neurons'], 
            activation=MLP_ARCHITECTURE['output_activation'],
            name='output_layer'
        )
    ])
    
    # Compile model
    model.compile(
        optimizer=OPTIMIZER,
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    return model

# ===================== EVALUATION =====================
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Đánh giá model - 4 metrics: Accuracy, Precision, Recall, F1"""
    
    # Predictions
    y_train_pred_prob = model.predict(X_train, batch_size=BATCH_SIZE, verbose=0)
    y_test_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE, verbose=0)
    
    y_train_pred = (y_train_pred_prob > 0.5).astype(int).flatten()
    y_test_pred = (y_test_pred_prob > 0.5).astype(int).flatten()
    
    # Train metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
    }
    
    # Test metrics
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
    }
    
    # Confusion matrix
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    return train_metrics, test_metrics, cm_test

# ===================== VISUALIZATION =====================
def plot_training_history(history_df, save_path):
    """Vẽ biểu đồ quá trình training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History - MLP', fontsize=16, fontweight='bold')
    
    epochs = history_df['epoch'].values
    
    # Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, history_df['train_accuracy'], 'b-o', label='Train', linewidth=2, markersize=6)
    ax1.plot(epochs, history_df['test_accuracy'], 'r-s', label='Test', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Over Epochs', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 Score
    ax2 = axes[0, 1]
    ax2.plot(epochs, history_df['train_f1'], 'b-o', label='Train', linewidth=2, markersize=6)
    ax2.plot(epochs, history_df['test_f1'], 'r-s', label='Test', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_title('F1 Score Over Epochs', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Precision
    ax3 = axes[1, 0]
    ax3.plot(epochs, history_df['train_precision'], 'b-o', label='Train', linewidth=2, markersize=6)
    ax3.plot(epochs, history_df['test_precision'], 'r-s', label='Test', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('Precision', fontweight='bold')
    ax3.set_title('Precision Over Epochs', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Recall
    ax4 = axes[1, 1]
    ax4.plot(epochs, history_df['train_recall'], 'b-o', label='Train', linewidth=2, markersize=6)
    ax4.plot(epochs, history_df['test_recall'], 'r-s', label='Test', linewidth=2, markersize=6)
    ax4.set_xlabel('Epoch', fontweight='bold')
    ax4.set_ylabel('Recall', fontweight='bold')
    ax4.set_title('Recall Over Epochs', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    console.print(f"[green]✓ Saved training history:[/green] {save_path}")

def plot_confusion_matrix(cm, metrics, save_path):
    """Vẽ confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malware'],
                yticklabels=['Benign', 'Malware'],
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - Best Model', fontsize=14, fontweight='bold', pad=20)
    
    # Percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.2f}%)', 
                   ha='center', va='center', fontsize=10, color='red')
    
    # Metrics box
    metrics_text = f"Accuracy:  {metrics['accuracy']:.6f}\n"
    metrics_text += f"Precision: {metrics['precision']:.6f}\n"
    metrics_text += f"Recall:    {metrics['recall']:.6f}\n"
    metrics_text += f"F1 Score:  {metrics['f1']:.6f}"
    
    ax.text(1.15, 0.5, metrics_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ===================== MAIN TRAINING =====================
def main():
    start_time = datetime.now()
    
    # Load data
    console.print("\n[bold cyan]═══ STEP 1: LOADING DATA ═══[/bold cyan]")
    X_full, y_full = load_full_dataset()
    
    # Split data
    console.print("\n[bold cyan]═══ STEP 2: TRAIN/TEST SPLIT ═══[/bold cyan]")
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, 
        train_size=TRAIN_RATIO, 
        random_state=RANDOM_STATE, 
        stratify=y_full
    )
    
    console.print(f"[green]✓ Train:[/green] {X_train.shape[0]:,} samples")
    console.print(f"[green]✓ Test:[/green]  {X_test.shape[0]:,} samples")
    
    # Normalize data
    console.print("\n[bold cyan]═══ STEP 3: DATA NORMALIZATION ═══[/bold cyan]")
    console.print("[cyan]Applying StandardScaler...[/cyan]")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    console.print(f"[green]✓ Saved scaler:[/green] {scaler_path}")
    
    del X_full, y_full, X_train, X_test
    gc.collect()
    
    # Build model
    console.print("\n[bold cyan]═══ STEP 4: BUILD MODEL ═══[/bold cyan]")
    model = build_mlp_model(input_dim=X_train_scaled.shape[1])
    
    console.print("\n[bold yellow]Model Architecture:[/bold yellow]")
    model.summary(print_fn=lambda x: console.print(x, style="dim"))
    
    total_params = model.count_params()
    console.print(f"\n[green]✓ Total Parameters:[/green] {total_params:,}")
    
    # Training with custom loop
    console.print("\n[bold cyan]═══ STEP 5: TRAINING ═══[/bold cyan]")
    
    history = []
    best_accuracy = 0
    best_epoch = 0
    no_improve_count = 0
    best_weights = None
    
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = datetime.now()
        
        console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
        console.print(f"[bold yellow]EPOCH {epoch}/{MAX_EPOCHS}[/bold yellow]")
        console.print(f"[bold yellow]{'='*80}[/bold yellow]")
        
        # Train one epoch
        console.print(f"[cyan]Training...[/cyan]")
        model.fit(
            X_train_scaled, y_train,
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=0
        )
        
        # Evaluate
        console.print(f"[cyan]Evaluating...[/cyan]")
        train_metrics, test_metrics, cm_test = evaluate_model(
            model, X_train_scaled, y_train, X_test_scaled, y_test
        )
        
        epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
        
        # Save history
        history.append({
            'epoch': epoch,
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'test_accuracy': test_metrics['accuracy'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall'],
            'test_f1': test_metrics['f1'],
            'time_minutes': epoch_time
        })
        
        # Display results
        results_table = Table(title=f"Epoch {epoch} Results", show_header=True, header_style="bold magenta")
        results_table.add_column("Metric", style="cyan", width=15)
        results_table.add_column("Train", justify="right", style="green", width=12)
        results_table.add_column("Test", justify="right", style="yellow", width=12)
        
        results_table.add_row("Accuracy", f"{train_metrics['accuracy']:.6f}", f"{test_metrics['accuracy']:.6f}")
        results_table.add_row("Precision", f"{train_metrics['precision']:.6f}", f"{test_metrics['precision']:.6f}")
        results_table.add_row("Recall", f"{train_metrics['recall']:.6f}", f"{test_metrics['recall']:.6f}")
        results_table.add_row("F1 Score", f"{train_metrics['f1']:.6f}", f"{test_metrics['f1']:.6f}")
        
        console.print(results_table)
        console.print(f"[dim]⏱️  Epoch time: {epoch_time:.1f} minutes[/dim]")
        
        # Check improvement
        current_accuracy = test_metrics['accuracy']
        improvement = current_accuracy - best_accuracy
        
        if improvement > MIN_DELTA:
            console.print(f"[bold green]✓ New best accuracy: {current_accuracy:.6f} (improved by {improvement*100:.4f}%)[/bold green]")
            best_accuracy = current_accuracy
            best_epoch = epoch
            no_improve_count = 0
            best_weights = model.get_weights()
            best_metrics = test_metrics
            best_cm = cm_test
            
        else:
            no_improve_count += 1
            console.print(f"[yellow]⚠️  No improvement (patience: {no_improve_count}/{PATIENCE})[/yellow]")
            
            if no_improve_count >= PATIENCE:
                console.print(f"[bold red]⏹️  Early stopping! No improvement for {PATIENCE} epochs.[/bold red]")
                console.print(f"[bold green]✓ Best accuracy: {best_accuracy:.6f} at epoch {best_epoch}[/bold green]")
                break
        
        gc.collect()
    
    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)
    
    # Save model
    console.print("\n[bold cyan]═══ STEP 6: SAVE MODEL ═══[/bold cyan]")
    
    # Save in Keras format (.keras)
    model_keras_path = os.path.join(OUTPUT_DIR, 'best_mlp_model.keras')
    model.save(model_keras_path)
    console.print(f"[green]✓ Saved model (Keras):[/green] {model_keras_path}")
    
    # Save in H5 format (legacy)
    model_h5_path = os.path.join(OUTPUT_DIR, 'best_mlp_model.h5')
    model.save(model_h5_path)
    console.print(f"[green]✓ Saved model (H5):[/green] {model_h5_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_best.png')
    plot_confusion_matrix(best_cm, best_metrics, cm_path)
    console.print(f"[green]✓ Saved confusion matrix:[/green] {cm_path}")
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(OUTPUT_DIR, 'training_history.csv')
    history_df.to_csv(history_csv, index=False)
    console.print(f"[green]✓ Saved history:[/green] {history_csv}")
    
    # Plot training history
    history_plot = os.path.join(OUTPUT_DIR, 'training_history.png')
    plot_training_history(history_df, history_plot)
    
    # Save training log
    log_path = os.path.join(OUTPUT_DIR, 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("MLP - MULTI-LAYER PERCEPTRON TRAINING LOG\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: EMBER 2018 (Malware Detection)\n")
        f.write(f"Train Samples: {len(y_train):,}\n")
        f.write(f"Test Samples: {len(y_test):,}\n")
        f.write(f"Features: {X_train_scaled.shape[1]}\n\n")
        
        f.write("MODEL ARCHITECTURE\n")
        f.write("-"*80 + "\n")
        f.write(f"Input Layer:    {MLP_ARCHITECTURE['input_neurons']} neurons\n")
        f.write(f"Hidden Layer 1: {MLP_ARCHITECTURE['hidden1_neurons']} neurons (ReLU + Dropout)\n")
        f.write(f"Hidden Layer 2: {MLP_ARCHITECTURE['hidden2_neurons']} neurons (ReLU + Dropout)\n")
        f.write(f"Hidden Layer 3: {MLP_ARCHITECTURE['hidden3_neurons']} neurons (ReLU + Dropout)\n")
        f.write(f"Output Layer:   {MLP_ARCHITECTURE['output_neurons']} neuron (Sigmoid)\n")
        f.write(f"Total Parameters: {total_params:,}\n\n")
        
        f.write("TRAINING PARAMETERS\n")
        f.write("-"*80 + "\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss Function: Binary Crossentropy\n")
        f.write(f"Max Epochs: {MAX_EPOCHS}\n")
        f.write(f"Early Stopping Patience: {PATIENCE}\n")
        f.write(f"Min Delta: {MIN_DELTA}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TRAINING HISTORY\n")
        f.write("="*80 + "\n\n")
        
        for row in history:
            f.write(f"EPOCH {row['epoch']}\n")
            f.write("-"*80 + "\n")
            
            f.write(f"Train Metrics:\n")
            f.write(f"  Accuracy:  {row['train_accuracy']:.6f}\n")
            f.write(f"  Precision: {row['train_precision']:.6f}\n")
            f.write(f"  Recall:    {row['train_recall']:.6f}\n")
            f.write(f"  F1 Score:  {row['train_f1']:.6f}\n\n")
            
            f.write(f"Test Metrics:\n")
            f.write(f"  Accuracy:  {row['test_accuracy']:.6f}\n")
            f.write(f"  Precision: {row['test_precision']:.6f}\n")
            f.write(f"  Recall:    {row['test_recall']:.6f}\n")
            f.write(f"  F1 Score:  {row['test_f1']:.6f}\n\n")
            
            f.write(f"Time: {row['time_minutes']:.1f} minutes\n")
            f.write("\n" + "="*80 + "\n\n")
        
        f.write("BEST MODEL\n")
        f.write("-"*80 + "\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Test Accuracy: {best_accuracy:.6f}\n")
        f.write(f"Best Test Precision: {best_metrics['precision']:.6f}\n")
        f.write(f"Best Test Recall: {best_metrics['recall']:.6f}\n")
        f.write(f"Best Test F1: {best_metrics['f1']:.6f}\n\n")
        
        total_time = (datetime.now() - start_time).total_seconds() / 60
        f.write(f"Total Training Time: {total_time:.1f} minutes ({total_time/60:.2f} hours)\n")
    
    console.print(f"[green]✓ Saved log:[/green] {log_path}")
    
    # Final summary
    console.print("\n" + "="*80, style="bold green")
    console.print("✅ TRAINING COMPLETED!", style="bold green", justify="center")
    console.print("="*80, style="bold green")
    
    summary_table = Table(title="📊 Final Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Total Epochs", str(len(history)))
    summary_table.add_row("Best Epoch", str(best_epoch))
    summary_table.add_row("Best Test Accuracy", f"{best_accuracy:.6f}")
    summary_table.add_row("Best Test Precision", f"{best_metrics['precision']:.6f}")
    summary_table.add_row("Best Test Recall", f"{best_metrics['recall']:.6f}")
    summary_table.add_row("Best Test F1", f"{best_metrics['f1']:.6f}")
    summary_table.add_row("Total Parameters", f"{total_params:,}")
    
    console.print(summary_table)
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    console.print(f"\n[bold green]⏱️  Total Time:[/bold green] {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    console.print(f"\n[bold green]💾 Models Saved:[/bold green]")
    console.print(f"   • best_mlp_model.keras")
    console.print(f"   • best_mlp_model.h5")
    console.print(f"   • scaler.pkl")
    console.print("="*80, style="bold green")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n\n[bold red]⚠️  Training interrupted![/bold red]")
    except Exception as e:
        console.print(f"\n\n[bold red]❌ Error:[/bold red] {str(e)}")
        import traceback
        traceback.print_exc()