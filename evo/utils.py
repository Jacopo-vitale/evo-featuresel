import numpy as np
import os
import random
import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

@dataclass
class Setup:
    """
    Setup class for the evolutionary algorithm experiment.
    """
    POP_SIZE: int = 50
    MUT_RATE: float = 1.0
    GENES: List[int] = field(default_factory=lambda: [0, 1])
    FILAMENT_LEN: int = 0
    DATA: Optional[Tuple[np.ndarray, np.ndarray]] = None
    LABELS: Optional[Tuple[np.ndarray, np.ndarray]] = None
    BITS: Dict[str, int] = field(default_factory=dict)
    DESCRIPTION: str = ""
    RANDOM_SEED: int = 42
    
    experiment_folder: str = 'experiment'
    project_prefix: str = ''
    
    def __post_init__(self):
        self.project_folder = os.path.join(
            self.experiment_folder,
            f"{self.project_prefix}{dt.datetime.now().strftime('%Y%m%d%H%M')}"
        )
        os.makedirs(self.experiment_folder, exist_ok=True)
        os.makedirs(self.project_folder, exist_ok=True)
        self.seed_all(self.RANDOM_SEED)

    def seed_all(self, seed: int):
        """
        Seeds all random number generators for reproducibility.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed=seed)

    def init_rng(self):
        # Kept for backward compatibility
        self.seed_all(self.RANDOM_SEED)

if __name__ == '__main__':
    evo_setup = Setup(POP_SIZE=500)
    print(f"Project folder: {evo_setup.project_folder}")
