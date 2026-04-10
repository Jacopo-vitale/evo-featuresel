import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np

SURFACE_BG = '#11151b'
PANEL_BG = '#171b22'
GRID_COLOR = '#4b5666'
TEXT_COLOR = '#d8dee9'
MUTED_TEXT = '#aab4c3'
BEST_LINE = '#f6c177'
AVG_LINE = '#8bd3dd'
BAR_COLOR = '#c4a7e7'
PIE_COLORS = ['#8bd3dd', '#f6c177', '#a6d189', '#eebebe', '#c4a7e7']


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
        self.fig.patch.set_facecolor(SURFACE_BG)

        self.ax_fitness = self.fig.add_subplot(2, 1, 1)
        self.ax_model = self.fig.add_subplot(2, 2, 3)
        self.ax_features = self.fig.add_subplot(2, 2, 4)

        self._style_axis(
            self.ax_fitness,
            "Evolution Progress (MCC)",
            xlabel="Generation",
            ylabel="MCC",
            grid=True
        )
        self._style_pie_axis()
        self._style_axis(self.ax_features, "Top Features Frequency", ylabel="Frequency")

        self.fig.tight_layout(pad=3.0)
        super().__init__(self.fig)

    def _style_axis(self, axis, title, xlabel=None, ylabel=None, grid=False):
        axis.set_facecolor(PANEL_BG)
        axis.set_title(title, fontsize=10, fontweight='bold', color=TEXT_COLOR)
        if xlabel is not None:
            axis.set_xlabel(xlabel, color=MUTED_TEXT)
        if ylabel is not None:
            axis.set_ylabel(ylabel, color=MUTED_TEXT)
        axis.tick_params(axis='x', colors=MUTED_TEXT)
        axis.tick_params(axis='y', colors=MUTED_TEXT)
        for spine in axis.spines.values():
            spine.set_color(GRID_COLOR)
        if grid:
            axis.grid(True, linestyle='--', alpha=0.35, color=GRID_COLOR)

    def _style_pie_axis(self):
        self.ax_model.set_facecolor(PANEL_BG)
        self.ax_model.set_title("Model Distribution", fontsize=10, fontweight='bold', color=TEXT_COLOR)
        self.ax_model.set_xticks([])
        self.ax_model.set_yticks([])
        for spine in self.ax_model.spines.values():
            spine.set_color(PANEL_BG)

    def plot_multi_data(self, generations, fitness_data, metric_data, metric_name="Accuracy"):
        """
        Plots fitness (best/avg) and an additional selected metric (best/avg).
        Uses twinx for different scales.
        fitness_data: (best_fitness, avg_fitness)
        metric_data: (best_metric, avg_metric)
        """
        best_fitness, avg_fitness = fitness_data
        best_metric, avg_metric = metric_data

        self.ax_fitness.clear()
        # Remove any existing twin axes
        if hasattr(self, 'ax_secondary'):
            self.ax_secondary.remove()
        
        self.ax_secondary = self.ax_fitness.twinx()
        
        self._style_axis(
            self.ax_fitness,
            f"Evolution Progress (Fitness & {metric_name})",
            xlabel="Generation",
            ylabel="Fitness (MCC - Penalty)",
            grid=True
        )
        self._style_secondary_axis(self.ax_secondary, ylabel=metric_name)

        if generations:
            # 1. Plot Fitness on primary axis
            ln1 = self.ax_fitness.plot(
                generations, best_fitness, 'o-', color=BEST_LINE, 
                linewidth=2.0, markersize=4, label='Best Fitness'
            )
            ln2 = self.ax_fitness.plot(
                generations, avg_fitness, '--', color=AVG_LINE, 
                alpha=0.7, linewidth=1.5, label='Avg Fitness'
            )

            # 2. Plot Selected Metric on secondary axis
            ln3 = self.ax_secondary.plot(
                generations, best_metric, 's-', color='#a6d189', 
                linewidth=1.8, markersize=4, label=f'Best {metric_name}'
            )
            ln4 = self.ax_secondary.plot(
                generations, avg_metric, ':', color='#eebebe', 
                alpha=0.7, linewidth=1.5, label=f'Avg {metric_name}'
            )

            # Combine legends
            lns = ln1 + ln2 + ln3 + ln4
            labs = [l.get_label() for l in lns]
            legend = self.ax_fitness.legend(lns, labs, loc='lower right', fontsize=7)
            legend.get_frame().set_facecolor(PANEL_BG)
            legend.get_frame().set_edgecolor(GRID_COLOR)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)

            # Auto-scale limits with padding (avoid fixed normalization)
            all_f = best_fitness + avg_fitness
            if all_f:
                min_f, max_f = min(all_f), max(all_f)
                padding_f = 0.05 * (max_f - min_f) if max_f != min_f else 0.05
                self.ax_fitness.set_ylim(min_f - padding_f, max_f + padding_f)
                
            all_m = best_metric + avg_metric
            if all_m:
                min_m, max_m = min(all_m), max(all_m)
                padding_m = 0.05 * (max_m - min_m) if max_m != min_m else 0.05
                self.ax_secondary.set_ylim(min_m - padding_m, max_m + padding_m)

        self.draw()

    def _style_secondary_axis(self, axis, ylabel=None):
        if ylabel is not None:
            axis.set_ylabel(ylabel, color='#a6d189')
        axis.tick_params(axis='y', colors='#a6d189')
        for spine in axis.spines.values():
            spine.set_color(GRID_COLOR)

    def plot_stats(self, stats):
        """
        Updates the pie and bar charts with population statistics.
        """
        self.ax_model.clear()
        self._style_pie_axis()
        model_counts = stats.get('model_counts', {})
        if model_counts:
            labels = list(model_counts.keys())
            sizes = list(model_counts.values())
            self.ax_model.pie(
                sizes,
                labels=labels,
                autopct='%1.1f%%',
                startangle=90,
                colors=PIE_COLORS[:len(labels)],
                textprops={'fontsize': 8, 'color': TEXT_COLOR},
                wedgeprops={'edgecolor': SURFACE_BG, 'linewidth': 1.1}
            )

        self.ax_features.clear()
        self._style_axis(self.ax_features, "Complexity Distribution", ylabel="Num Individuals", xlabel="Selected Features")
        complexity_dist = stats.get('complexity_dist', [])
        
        if complexity_dist:
            # Create a histogram of the number of features selected
            n_features = stats.get('n_features', max(complexity_dist) if complexity_dist else 100)
            
            # User specifically requested setting the number of bins equal to the number of features
            bins = max(1, n_features)
            
            self.ax_features.hist(
                complexity_dist, 
                bins=bins, 
                color=BAR_COLOR, 
                edgecolor='#eed7ff', 
                alpha=0.8,
                label='Individuals'
            )
            
            # Add a vertical line for average
            avg_c = np.mean(complexity_dist)
            self.ax_features.axvline(avg_c, color=BEST_LINE, linestyle='--', linewidth=1.5, label='Avg')
            
            self.ax_features.legend(fontsize=7, loc='upper right')
            self.ax_features.tick_params(axis='x', labelsize=8, colors=MUTED_TEXT)
            self.ax_features.tick_params(axis='y', labelsize=8, colors=MUTED_TEXT)

        self.fig.tight_layout(pad=3.0)
        self.draw()
