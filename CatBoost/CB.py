# catboost_optimized_params.py
# CATBOOST - Optimized Parameters for Malware Detection

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
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
OUTPUT_DIR = 'D:/Final_keylogger_ML_2/CatBoost'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Train/Test Split
TRAIN_RATIO = 0.8
RANDOM_STATE = 42

# OPTIMIZED PARAMETERS (tinh chỉnh cho Malware Detection)
OPTIMIZED_PARAMS = {
    # Core parameters
    'loss_function': 'Logloss',           # Binary classification
    'eval_metric': 'Accuracy',            # Metric chính để tối ưu
    'custom_metric': ['F1', 'Precision', 'Recall'],  # Chỉ 3 metrics: F1, Precision, Recall
    'learning_rate': 0.05,
    'iterations': 5000,
    'depth': 6,
    'l2_leaf_reg': 3,
    
    # Data processing
    'border_count': 254,
    'rsm': 0.8,                           # Random subspace method
    
    # Randomization
    'bagging_temperature': 0.5,
    'random_strength': 1.0,
    
    # Early stopping (chỉ dùng early_stopping_rounds)
    'early_stopping_rounds': 150,         # Chỉ dùng 1 trong 2: od_wait HOẶC early_stopping_rounds
    
    # System
    'verbose': 100,
    'thread_count': 4,
    'random_seed': RANDOM_STATE,
    'use_best_model': True,
    'allow_writing_files': False,
}

console.print("="*80, style="bold cyan")
console.print("CATBOOST - OPTIMIZED PARAMETERS", style="bold cyan", justify="center")
console.print("="*80, style="bold cyan")
console.print(f"[yellow]Dataset:[/yellow] EMBER 2018 (Malware Detection)")
console.print(f"[yellow]Model:[/yellow] CatBoost Gradient Boosting")
console.print(f"[yellow]Primary Metric:[/yellow] Accuracy")
console.print(f"[yellow]Additional Metrics:[/yellow] F1, Precision, Recall")
console.print(f"[yellow]Optimized Hyperparameters:[/yellow]")
for k, v in OPTIMIZED_PARAMS.items():
    if k not in ['random_seed', 'verbose', 'thread_count', 'use_best_model', 'allow_writing_files', 'custom_metric']:
        console.print(f"  • {k}: {v}")
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
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Đánh giá model - 4 metrics: Accuracy, Precision, Recall, F1"""
    
    # Predictions
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()
    
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
    fig.suptitle('Training History - CatBoost', fontsize=16, fontweight='bold')
    
    iterations = history_df['iteration'].values
    
    # Accuracy
    ax1 = axes[0, 0]
    if 'learn_Accuracy' in history_df.columns and 'test_Accuracy' in history_df.columns:
        ax1.plot(iterations, history_df['learn_Accuracy'], 'b-', label='Train', linewidth=2)
        ax1.plot(iterations, history_df['test_Accuracy'], 'r-', label='Test', linewidth=2)
        ax1.set_xlabel('Iterations', fontweight='bold')
        ax1.set_ylabel('Accuracy', fontweight='bold')
        ax1.set_title('Accuracy Over Iterations', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # F1 Score
    ax2 = axes[0, 1]
    if 'learn_F1' in history_df.columns and 'test_F1' in history_df.columns:
        ax2.plot(iterations, history_df['learn_F1'], 'b-', label='Train', linewidth=2)
        ax2.plot(iterations, history_df['test_F1'], 'r-', label='Test', linewidth=2)
        ax2.set_xlabel('Iterations', fontweight='bold')
        ax2.set_ylabel('F1 Score', fontweight='bold')
        ax2.set_title('F1 Score Over Iterations', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Precision
    ax3 = axes[1, 0]
    if 'learn_Precision' in history_df.columns and 'test_Precision' in history_df.columns:
        ax3.plot(iterations, history_df['learn_Precision'], 'b-', label='Train', linewidth=2)
        ax3.plot(iterations, history_df['test_Precision'], 'r-', label='Test', linewidth=2)
        ax3.set_xlabel('Iterations', fontweight='bold')
        ax3.set_ylabel('Precision', fontweight='bold')
        ax3.set_title('Precision Over Iterations', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Recall
    ax4 = axes[1, 1]
    if 'learn_Recall' in history_df.columns and 'test_Recall' in history_df.columns:
        ax4.plot(iterations, history_df['learn_Recall'], 'b-', label='Train', linewidth=2)
        ax4.plot(iterations, history_df['test_Recall'], 'r-', label='Test', linewidth=2)
        ax4.set_xlabel('Iterations', fontweight='bold')
        ax4.set_ylabel('Recall', fontweight='bold')
        ax4.set_title('Recall Over Iterations', fontweight='bold')
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
    
    # Create CatBoost Pool objects
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)
    
    del X_full, y_full
    gc.collect()
    
    # Training
    console.print("\n[bold cyan]═══ STEP 3: TRAINING CATBOOST ═══[/bold cyan]")
    console.print(f"[cyan]Training CatBoost with optimized parameters...[/cyan]")
    console.print(f"[yellow]⚠️  This may take a while. Please wait...[/yellow]\n")
    
    # Initialize model
    model = CatBoostClassifier(**OPTIMIZED_PARAMS)
    
    # Train with evaluation
    model.fit(
        train_pool,
        eval_set=test_pool,
        plot=False
    )
    
    training_time = (datetime.now() - start_time).total_seconds() / 60
    
    console.print(f"\n[green]✓ Training completed![/green]")
    console.print(f"[green]✓ Total iterations:[/green] {model.tree_count_}")
    console.print(f"[green]✓ Training time:[/green] {training_time:.1f} minutes")
    
    # Evaluate
    console.print("\n[bold cyan]═══ STEP 4: EVALUATION ═══[/bold cyan]")
    console.print(f"[cyan]Evaluating model...[/cyan]")
    
    train_metrics, test_metrics, cm_test = evaluate_model(
        model, X_train, y_train, X_test, y_test
    )
    
    # Display results
    results_table = Table(title="Final Results", show_header=True, header_style="bold magenta")
    results_table.add_column("Metric", style="cyan", width=15)
    results_table.add_column("Train", justify="right", style="green", width=12)
    results_table.add_column("Test", justify="right", style="yellow", width=12)
    
    results_table.add_row("Accuracy", f"{train_metrics['accuracy']:.6f}", f"{test_metrics['accuracy']:.6f}")
    results_table.add_row("Precision", f"{train_metrics['precision']:.6f}", f"{test_metrics['precision']:.6f}")
    results_table.add_row("Recall", f"{train_metrics['recall']:.6f}", f"{test_metrics['recall']:.6f}")
    results_table.add_row("F1 Score", f"{train_metrics['f1']:.6f}", f"{test_metrics['f1']:.6f}")
    
    console.print(results_table)
    
    # Save model
    console.print("\n[bold cyan]═══ STEP 5: SAVE MODEL ═══[/bold cyan]")
    
    model_pkl_path = os.path.join(OUTPUT_DIR, 'best_catboost_model.pkl')
    joblib.dump(model, model_pkl_path)
    console.print(f"[green]✓ Saved best model (PKL):[/green] {model_pkl_path}")
    
    # Save confusion matrix
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_best.png')
    plot_confusion_matrix(cm_test, test_metrics, cm_path)
    console.print(f"[green]✓ Saved confusion matrix:[/green] {cm_path}")
    
    # Save training history
    try:
        train_history = model.get_evals_result()
        if train_history:
            # Extract training history
            history_data = {'iteration': []}
            
            # Get iteration numbers
            for dataset in ['learn', 'validation']:
                if dataset in train_history:
                    for metric_name in train_history[dataset].keys():
                        history_data['iteration'] = list(range(len(train_history[dataset][metric_name])))
                        break
                    break
            
            # Add all metrics
            for dataset in ['learn', 'validation']:
                if dataset in train_history:
                    dataset_prefix = 'learn' if dataset == 'learn' else 'test'
                    for metric_name, values in train_history[dataset].items():
                        col_name = f'{dataset_prefix}_{metric_name}'
                        history_data[col_name] = values
            
            history_df = pd.DataFrame(history_data)
            history_csv = os.path.join(OUTPUT_DIR, 'training_history.csv')
            history_df.to_csv(history_csv, index=False)
            console.print(f"[green]✓ Saved training history:[/green] {history_csv}")
            
            # Plot training history
            history_plot = os.path.join(OUTPUT_DIR, 'training_history.png')
            plot_training_history(history_df, history_plot)
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not save training history: {str(e)}[/yellow]")
    
    # Save feature importance
    try:
        feature_importance = model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(feature_importance))],
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        importance_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        console.print(f"[green]✓ Saved feature importance:[/green] {importance_path}")
        
        # Plot top 30 features
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(30)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance', fontweight='bold')
        plt.title('Top 30 Feature Importance', fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        importance_plot = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(importance_plot, dpi=150, bbox_inches='tight')
        plt.close()
        console.print(f"[green]✓ Saved importance plot:[/green] {importance_plot}")
    except Exception as e:
        console.print(f"[yellow]⚠️  Could not save feature importance: {str(e)}[/yellow]")
    
    # Save training log
    log_path = os.path.join(OUTPUT_DIR, 'training_log.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CATBOOST - TRAINING LOG\n")
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
            f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        f.write("TRAINING RESULTS\n")
        f.write("-"*80 + "\n")
        f.write(f"Total Trees: {model.tree_count_}\n")
        f.write(f"Training Time: {training_time:.1f} minutes ({training_time/60:.2f} hours)\n\n")
        
        f.write("TRAIN METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"  Accuracy:  {train_metrics['accuracy']:.6f}\n")
        f.write(f"  Precision: {train_metrics['precision']:.6f}\n")
        f.write(f"  Recall:    {train_metrics['recall']:.6f}\n")
        f.write(f"  F1 Score:  {train_metrics['f1']:.6f}\n\n")
        
        f.write("TEST METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.6f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.6f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.6f}\n")
        f.write(f"  F1 Score:  {test_metrics['f1']:.6f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-"*80 + "\n")
        f.write(f"  True Negatives:  {cm_test[0, 0]:,}\n")
        f.write(f"  False Positives: {cm_test[0, 1]:,}\n")
        f.write(f"  False Negatives: {cm_test[1, 0]:,}\n")
        f.write(f"  True Positives:  {cm_test[1, 1]:,}\n")
    
    console.print(f"[green]✓ Saved log:[/green] {log_path}")
    
    # Final summary
    console.print("\n" + "="*80, style="bold green")
    console.print("✅ TRAINING COMPLETED!", style="bold green", justify="center")
    console.print("="*80, style="bold green")
    
    summary_table = Table(title="📊 Final Summary", show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="yellow")
    
    summary_table.add_row("Total Trees", str(model.tree_count_))
    summary_table.add_row("Test Accuracy", f"{test_metrics['accuracy']:.6f}")
    summary_table.add_row("Test Precision", f"{test_metrics['precision']:.6f}")
    summary_table.add_row("Test Recall", f"{test_metrics['recall']:.6f}")
    summary_table.add_row("Test F1 Score", f"{test_metrics['f1']:.6f}")
    
    console.print(summary_table)
    
    total_time = (datetime.now() - start_time).total_seconds() / 60
    console.print(f"\n[bold green]⏱️  Total Time:[/bold green] {total_time:.1f} minutes ({total_time/60:.2f} hours)")
    console.print(f"\n[bold green]💾 Best Model Saved:[/bold green] best_catboost_model.pkl")
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