# xgboost_optimized_params.py
# XGBOOST - Optimized Parameters for Malware Detection

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, 
                            confusion_matrix, precision_score, recall_score)
from sklearn.model_selection import train_test_split
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

console = Console()

# ===================== CẤU HÌNH =====================
INPUT_DIR = 'D:/Final_keylogger_ML_2/processed_features_cleaned'
OUTPUT_DIR = 'D:/Final_keylogger_ML_2/XGBoost'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Train/Test Split
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# OPTIMIZED PARAMETERS (tối ưu cho Malware Detection với high-dimensional data)
OPTIMIZED_PARAMS = {
    # Core parameters
    'objective': 'binary:logistic',      # Binary classification
    'eval_metric': 'error',              # 1 - Accuracy (error rate)
    'learning_rate': 0.05,               # Tốc độ học (eta)
    'max_depth': 8,                      # Độ sâu cây
    'min_child_weight': 3,               # Minimum sum of instance weight in a child
    
    # Regularization
    'gamma': 0.1,                        # Minimum loss reduction for split
    'alpha': 0.1,                        # L1 regularization
    'lambda': 1.0,                       # L2 regularization (lambda)
    
    # Sampling
    'subsample': 0.8,                    # Row sampling
    'colsample_bytree': 0.8,             # Column sampling per tree
    'colsample_bylevel': 0.8,            # Column sampling per level
    
    # System
    'tree_method': 'hist',               # Fast histogram algorithm
    'random_state': RANDOM_STATE,
    'n_jobs': 4,
}

# Training Configuration
NUM_BOOST_ROUND = 100      # Số trees thêm mỗi epoch
MAX_EPOCHS = 60            # Số epoch tối đa
PATIENCE = 10              # Early stopping patience
MIN_IMPROVEMENT = 0.0001   # Cải thiện tối thiểu (0.01%)

console.print("="*80, style="bold cyan")
console.print("XGBOOST - OPTIMIZED PARAMETERS", style="bold cyan", justify="center")
console.print("="*80, style="bold cyan")
console.print(f"[yellow]Dataset:[/yellow] EMBER 2018 (Malware Detection)")
console.print(f"[yellow]Model:[/yellow] XGBoost Gradient Boosting")
console.print(f"[yellow]Strategy:[/yellow] Incremental training until saturation")
console.print(f"[yellow]Optimized Hyperparameters:[/yellow]")
for k, v in OPTIMIZED_PARAMS.items():
    if k not in ['random_state', 'n_jobs', 'objective', 'eval_metric']:
        console.print(f"  • {k}: {v}")
console.print(f"[yellow]Boost Rounds per Epoch:[/yellow] {NUM_BOOST_ROUND}")
console.print(f"[yellow]Max Epochs:[/yellow] {MAX_EPOCHS}")
console.print(f"[yellow]Early Stopping Patience:[/yellow] {PATIENCE} epochs")
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

# ===================== EVALUATION =====================
def evaluate_model(model, dtrain, dtest, y_train, y_test):
    """Đánh giá model - 4 metrics: Accuracy, Precision, Recall, F1"""
    
    # Predictions
    y_train_pred = (model.predict(dtrain) > 0.5).astype(int)
    y_test_pred = (model.predict(dtest) > 0.5).astype(int)
    
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
    fig.suptitle('Training History - XGBoost (Optimized Parameters)', fontsize=16, fontweight='bold')
    
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
    
    # Create DMatrix (XGBoost's optimized data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    del X_full, y_full
    gc.collect()
    
    # Iterative Training
    console.print("\n[bold cyan]═══ STEP 3: ITERATIVE TRAINING ═══[/bold cyan]")
    
    history = []
    best_accuracy = 0
    best_epoch = 0
    no_improve_count = 0
    best_model = None
    best_metrics = None
    best_cm = None
    current_model = None
    
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_start = datetime.now()
        
        console.print(f"\n[bold yellow]{'='*80}[/bold yellow]")
        console.print(f"[bold yellow]EPOCH {epoch}/{MAX_EPOCHS}[/bold yellow]")
        console.print(f"[bold yellow]{'='*80}[/bold yellow]")
        
        # Train model
        console.print(f"[cyan]Training XGBoost with optimized parameters...[/cyan]")
        
        if current_model is None:
            # First epoch - train from scratch
            current_model = xgb.train(
                OPTIMIZED_PARAMS,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                evals=[(dtest, 'test')],
                verbose_eval=False
            )
        else:
            # Continue training from previous model
            current_model = xgb.train(
                OPTIMIZED_PARAMS,
                dtrain,
                num_boost_round=NUM_BOOST_ROUND,
                xgb_model=current_model,
                evals=[(dtest, 'test')],
                verbose_eval=False
            )
        
        # Evaluate
        console.print(f"[cyan]Evaluating...[/cyan]")
        train_metrics, test_metrics, cm_test = evaluate_model(
            current_model, dtrain, dtest, y_train, y_test
        )
        
        epoch_time = (datetime.now() - epoch_start).total_seconds() / 60
        
        # Save history
        history.append({
            'epoch': epoch,
            'total_trees': current_model.num_boosted_rounds(),
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
        console.print(f"[dim]🌳 Total trees: {current_model.num_boosted_rounds()}[/dim]")
        console.print(f"[dim]⏱️  Epoch time: {epoch_time:.1f} minutes[/dim]")
        
        # Check improvement
        current_accuracy = test_metrics['accuracy']
        improvement = current_accuracy - best_accuracy
        
        if improvement > MIN_IMPROVEMENT:
            console.print(f"[bold green]✓ New best accuracy: {current_accuracy:.6f} (improved by {improvement*100:.4f}%)[/bold green]")
            best_accuracy = current_accuracy
            best_epoch = epoch
            no_improve_count = 0
            best_model = current_model
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
    
    # Save ONLY the best model
    console.print("\n[bold cyan]═══ STEP 4: SAVE BEST MODEL ═══[/bold cyan]")
    
    if best_model is not None:
        # Save best model (PKL only)
        model_pkl_path = os.path.join(OUTPUT_DIR, 'best_xgboost_model.pkl')
        joblib.dump(best_model, model_pkl_path)
        console.print(f"[green]✓ Saved best model (PKL):[/green] {model_pkl_path}")
        
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
    
    # Save feature importance
    if best_model is not None:
        importance_dict = best_model.get_score(importance_type='gain')
        
        # Convert to DataFrame (giữ nguyên tên features như XGBoost trả về)
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance_dict.items()
        ]).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        console.print(f"[green]✓ Saved feature importance:[/green] {importance_path}")
        
        # Plot top 30 features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(30)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance (Gain)', fontweight='bold')
        plt.title('Top 30 Feature Importance', fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        importance_plot = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(importance_plot, dpi=150, bbox_inches='tight')
        plt.close()
        console.print(f"[green]✓ Saved importance plot:[/green] {importance_plot}")
    
    # Save training log
    log_path = os.path.join(OUTPUT_DIR, 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("XGBOOST - ITERATIVE TRAINING LOG\n")
        f.write("="*80 + "\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: EMBER 2018 (Malware Detection)\n")
        f.write(f"Train Samples: {len(y_train):,}\n")
        f.write(f"Test Samples: {len(y_test):,}\n")
        f.write(f"Features: {X_train.shape[1]}\n\n")
        
        f.write("OPTIMIZED PARAMETERS\n")
        f.write("-"*80 + "\n")
        for k, v in OPTIMIZED_PARAMS.items():
            if k not in ['random_state', 'n_jobs']:
                f.write(f"  {k}: {v}\n")
        f.write(f"\nBoost Rounds per Epoch: {NUM_BOOST_ROUND}\n")
        f.write(f"Max Epochs: {MAX_EPOCHS}\n")
        f.write(f"Early Stopping Patience: {PATIENCE}\n")
        f.write(f"Min Improvement: {MIN_IMPROVEMENT*100:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("TRAINING HISTORY\n")
        f.write("="*80 + "\n\n")
        
        for row in history:
            f.write(f"EPOCH {row['epoch']}\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Trees: {row['total_trees']}\n\n")
            
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
        if best_model is not None:
            f.write(f"Total Trees: {best_model.num_boosted_rounds()}\n")
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
    
    if best_model is not None:
        best_row = history_df[history_df['epoch'] == best_epoch].iloc[0]
        summary_table.add_row("Best Test Precision", f"{best_metrics['precision']:.6f}")
        summary_table.add_row("Best Test Recall", f"{best_metrics['recall']:.6f}")
        summary_table.add_row("Best Test F1", f"{best_metrics['f1']:.6f}")
        summary_table.add_row("Total Trees", str(best_model.num_boosted_rounds()))
    
    console.print(summary_table)
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    console.print(f"\n[bold green]⏱️  Total Time:[/bold green] {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    console.print(f"\n[bold green]💾 Best Model Saved:[/bold green] best_xgboost_model.pkl")
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