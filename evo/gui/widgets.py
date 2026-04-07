import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

class MplCanvas(FigureCanvasQTAgg):
    """
    A Matplotlib canvas integrated in a PySide6 widget.
    Displays three plots:
    1. Fitness Trend (Line)
    2. Model Distribution (Pie)
    3. Feature Selection Frequency (Bar)
    """
    def __init__(self, parent=None, width=5, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#f0f0f0')
        
        # 1. Top: Fitness Plot
        self.ax_fitness = self.fig.add_subplot(2, 1, 1)
        self.ax_fitness.set_title("Evolution Progress (MCC)", fontsize=10, fontweight='bold')
        self.ax_fitness.set_xlabel("Generation")
        self.ax_fitness.set_ylabel("MCC")
        self.ax_fitness.grid(True, linestyle='--', alpha=0.6)
        
        # 2. Bottom Left: Model Distribution (Pie)
        self.ax_model = self.fig.add_subplot(2, 2, 3)
        self.ax_model.set_title("Model Distribution", fontsize=10, fontweight='bold')
        
        # 3. Bottom Right: Feature Frequency (Bar)
        self.ax_features = self.fig.add_subplot(2, 2, 4)
        self.ax_features.set_title("Top Features Frequency", fontsize=10, fontweight='bold')
        
        self.fig.tight_layout(pad=3.0)
        super().__init__(self.fig)

    def plot_data(self, generations, best_fitness, avg_fitness):
        self.ax_fitness.clear()
        self.ax_fitness.set_title("Evolution Progress (MCC)", fontsize=10, fontweight='bold')
        self.ax_fitness.set_xlabel("Generation")
        self.ax_fitness.set_ylabel("MCC")
        self.ax_fitness.grid(True, linestyle='--', alpha=0.6)
        
        if generations:
            self.ax_fitness.plot(generations, best_fitness, 'o-', color='#e74c3c', linewidth=2, label='Best Individual')
            self.ax_fitness.plot(generations, avg_fitness, '--', color='#2980b9', alpha=0.7, label='Pop. Average')
            self.ax_fitness.legend(loc='lower right', fontsize=8)
            
            # Dynamic Y limit with padding
            all_f = best_fitness + avg_fitness
            if all_f:
                min_f, max_f = min(all_f), max(all_f)
                padding = 0.05
                self.ax_fitness.set_ylim(max(-1.0, min_f - padding), min(1.0, max_f + padding))
        
        self.draw()

    def plot_stats(self, stats):
        """
        Updates the pie and bar charts with population statistics.
        """
        # --- Model Distribution (Pie) ---
        self.ax_model.clear()
        self.ax_model.set_title("Model Distribution", fontsize=10, fontweight='bold')
        model_counts = stats.get('model_counts', {})
        if model_counts:
            labels = list(model_counts.keys())
            sizes = list(model_counts.values())
            colors = ['#3498db', '#9b59b6', '#2ecc71', '#f1c40f', '#e67e22']
            self.ax_model.pie(sizes, labels=labels, autopct='%1.1f%%', 
                             startangle=90, colors=colors[:len(labels)],
                             textprops={'fontsize': 8})
        
        # --- Feature Frequency (Bar) ---
        self.ax_features.clear()
        self.ax_features.set_title("Top Features Frequency", fontsize=10, fontweight='bold')
        feat_freq = stats.get('feat_freq', [])
        if feat_freq:
            # Show only top 15 features
            top_n = min(len(feat_freq), 15)
            # feat_freq is expected as list of (feat_idx, count)
            indices = [str(f[0]) for f in feat_freq[:top_n]]
            counts = [f[1] for f in feat_freq[:top_n]]
            
            self.ax_features.bar(indices, counts, color='#1abc9c')
            self.ax_features.tick_params(axis='x', rotation=45, labelsize=7)
            self.ax_features.tick_params(axis='y', labelsize=8)
            self.ax_features.set_ylabel("Frequency", fontsize=8)
        
        self.fig.tight_layout(pad=3.0)
        self.draw()
