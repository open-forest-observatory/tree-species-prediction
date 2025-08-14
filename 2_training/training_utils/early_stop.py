from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any
import numpy as np

@dataclass
class EarlyStopper:
    # configureable
    patience: int = 0 # early stop disabled if 0 
    improvement_margin: float = 0.0   
    monitor_metric: Literal['f1', 'recall', 'precision', 'accuracy', 'loss'] = "f1"
    objective: Literal["max","min"] = "max"  

    # runtime state
    best_value: float = field(default_factory=lambda: -np.inf) # auto updates to pos inf if objective is 'min'
    bad_epochs: int = 0
    stopped: bool = False
    enabled: bool = field(init=False, repr=True)  # derived from patience>0

    def __post_init__(self):
        self.enabled = self.patience > 0
        if self.objective == "min":
            self.best_value = np.inf

    def step(self, metrics: Dict[str, Any]) -> Optional[tuple[bool, bool]]:
        """Returns (stop_now, is_best) if enabled"""
        if not self.enabled:
            return

        assert self.monitor_metric in metrics, f"'{self.monitor}' not in metrics: {list(metrics.keys())}"
        value = float(metrics[self.monitor_metric])
        is_improved = (
            (value > self.best_value + self.improvement_margin) if self.objective == "max"
            else (value < self.best_value - self.improvement_margin)
        )
        if is_improved:
            self.best_value = value
            self.bad_epochs = 0
            return (False, True)
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.stopped = True
            return (self.stopped, False)
