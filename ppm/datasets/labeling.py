import numpy as np
import pandas as pd
from typing import List
    


def verify_bpi15_ltl_rule(trace: List[str]):
    """
    Checks the rule: G(send_receipt -> F(retrieve_data))
    Returns True if the trace satisfies the rule, False otherwise.
    """
    # 1. Find all indices where the 'trigger' occurs
    trigger = "send confirmation receipt"
    target = "retrieve missing data"
    
    # Get positions of all trigger events
    trigger_indices = [i for i, event in enumerate(trace) if event == trigger]
    
    # 2. For every trigger found, check if the target exists later in the trace
    for idx in trigger_indices:
        # Look at the slice of the trace starting from this event to the end
        future_events = trace[idx:] 
        
        # Check if any event in the future matches the target
        target_found = any(event == target for event in future_events)
        
        if not target_found:
            # If even one trigger lacks an eventual target, the 'Global' rule fails
            return 0
            
    # If we never returned False, the rule is satisfied (or the trigger never occurred)
    return 1