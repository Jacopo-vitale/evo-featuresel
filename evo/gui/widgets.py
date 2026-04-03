import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    """
    A Matplotlib canvas integrated into a Qt Widget.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Set a slightly darker background for the figure to match modern UIs
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#f0f0f0')
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        
        # Optimize margins
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.12)
        
        self.axes.set_title("Evolutionary Fitness Trend", fontsize=10, fontweight='bold')
        self.axes.set_xlabel("Generation", fontsize=9)
        self.axes.set_ylabel("MCC", fontsize=9)
        
        # Initialize empty lines for performance
        self.line_best, = self.axes.plot([], [], label="Best Fitness", marker='o', markersize=4, color='#2ecc71', linewidth=2)
        self.line_avg, = self.axes.plot([], [], label="Avg Fitness", marker='x', markersize=4, color='#3498db', linestyle='--', linewidth=1.5)
        
        self.axes.legend(fontsize=8, loc='lower right')
        self.axes.grid(True, linestyle=':', alpha=0.4)
        self.axes.tick_params(labelsize=8)
        
    def plot_data(self, generations, best_fitness, avg_fitness):
        """
        Updates the canvas with new fitness data efficiently.
        """
        # Update line data instead of clearing and redrawing
        self.line_best.set_data(generations, best_fitness)
        self.line_avg.set_data(generations, avg_fitness)
        
        # Adjust limits if needed
        if generations:
            self.axes.set_xlim(0, max(10, max(generations)))
            
            # Auto-scale Y axis for MCC
            all_f = best_fitness + avg_fitness
            if all_f:
                min_f, max_f = min(all_f), max(all_f)
                padding = 0.05
                self.axes.set_ylim(max(-1.0, min_f - padding), min(1.0, max_f + padding))
        
        self.draw()
