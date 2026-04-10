import sys
from PySide6.QtWidgets import QApplication
from evo.gui.main_window import MainWindow

app = QApplication(sys.argv)
window = MainWindow()

# Simulate reset
window._on_reset()

# Simulate user changing fields
window.description.setText('My New Test')
window.penalty.setValue(0.55)
window.pop_size.setValue(999)
window.train_path.setText("data/dataset.csv")

# Start run
window._on_start()

# Check what was captured
print('Description in worker:', window.worker.params['description'])
print('Penalty in worker:', window.worker.params['penalty_factor'])
print('Pop size in worker:', window.worker.params['pop_size'])
print('Train path in worker:', window.worker.params['train_path'])
