import math
import copy
from typing import List
from collections import defaultdict


def add_start_end(t):
    """
    Add start and end markers to a given trace (tuple). We use this so we can just use a graph instead of having to define initial probabilities.

    Parameters:
    - t (tuple): The trace (tuple) to which start and end markers will be added.

    Returns:
    - tuple: The modified tuple with start and end markers added.
    """
    return ('BOS',) + t + ('EOS',)


def get_probability(activity_counts, edge_counts, trace):
    """
    Calculate the probability of a given trace to be replayed by a graph.

    Parameters:
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.
    - trace (tuple): The trace for which the probability needs to be calculated.

    Returns:
    - float: The probability of the given trace in the graph.
    """
    total_probability = 1.0
    trace_with_start_end = add_start_end(trace)
    for i in range(len( trace_with_start_end) - 1): 
        current_activity =  trace_with_start_end[i]
        next_activity =  trace_with_start_end[i + 1]      
        # Calculate the probability of taking the edge from current_activity to next_activity
        outgoing_edges = sum(edge_counts.get((current_activity, other_element), 0) for other_element in activity_counts.keys())
        if outgoing_edges > 0:
            edge_probability = edge_counts.get((current_activity, next_activity), 0) / outgoing_edges
            total_probability *= edge_probability
        else:
            # If no outgoing edges, set probability to 0
            total_probability = 0.0
            break
    return total_probability


def get_dfg(variant_log):
    """
    Creates a dfg by calculating the counts of activities and edges in the variant log.

    Args:
        variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.

    Returns:
        activity_counts (dict): A dictionary where the keys are activities and the values are the total counts of each activity in the variant log.
        edge_counts (dict): A dictionary where the keys are pairs of activities representing edges and the values are the total counts of each edge in the variant log.
    """
    logdummy = copy.deepcopy(variant_log)
    variant_log_with_start_end = {add_start_end(key): value for key, value in logdummy.items()}
    activity_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    
    for variant, occurrence in variant_log_with_start_end.items():
        for i in range(len(variant) - 1):
            current_activity, next_activity = variant[i], variant[i + 1]
            activity_counts[current_activity] += occurrence
            edge_counts[(current_activity, next_activity)] += occurrence
        #count last activity
        activity_counts[variant[-1]] += occurrence
    
    return dict(activity_counts), dict(edge_counts)


def update_dfg(activity_counts, edge_counts, new_trace, occurrence):
    """
    Update the Directly-Follows Graph (DFG) with a new trace.

    Parameters:
    - activity_counts (defaultdict): A dictionary containing the counts of each activity in the DFG.
    - edge_counts (defaultdict): A dictionary containing the counts of each edge in the DFG.
    - new_trace (tuple): The new trace to be added to the DFG.
    - occurrence (int): The number of times the new trace occurred.

    Returns:
    - activity_counts (defaultdict): The updated activity counts after adding the new trace.
    - edge_counts (defaultdict): The updated edge counts after adding the new trace.
    """
    new_trace_with_start_end = add_start_end(new_trace)
    for i in range(len(new_trace_with_start_end) - 1):
        current_activity = new_trace_with_start_end[i]
        next_activity = new_trace_with_start_end[i + 1]
        # Add new activity and edge if they do not exist
        # else add occurence to counts
        if current_activity not in activity_counts:
            activity_counts[current_activity] = occurrence
        else:
            activity_counts[current_activity] += occurrence
        if (current_activity, next_activity) not in edge_counts:
            edge_counts[(current_activity, next_activity)] = occurrence
        else:
            edge_counts[(current_activity, next_activity)] += occurrence

    #add end count
    activity_counts[new_trace_with_start_end[-1]] += occurrence

    return activity_counts, edge_counts


def get_ER(variant_log, activity_counts, edge_counts):
    """
    Calculate the Entropic Relevance (ER) value for a given variant log, activity counts, and edge counts (last two together dfg).
    #! This is the average ER over all traces in the log

    Parameters:
    - variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.

    Returns:
    - float: The ER value for the given variant log, activity counts, and edge counts.
    """
    ER_sum = 0.0
    total_occurences = 0
    for variant, occurrence in variant_log.items():
        prob = get_probability(activity_counts, edge_counts, variant)
        #use this for the logs where probabilities get too small
        prob = max(prob, 1e-10)
        ER_sum += (-math.log(prob, 2))*occurrence
        total_occurences += occurrence
    ER = ER_sum/total_occurences
    return ER


def get_ER_normalized(variant_log, activity_counts, edge_counts):
    """
    Calculate the normalized Entropic Relevance (ER) value for a given variant log, activity counts, and edge counts. it corrects the ER function by adjusting it to not 
    take into account the inherent decrease in probability introduced by loops. We therefore deduct the ER score of each trace, on a dfg mined on only that trace itself
    We might want to use this to make sure  that when taking the highest pairwise ER distance we do no necessarily prefer traces with loops. 

    Parameters:
    - variant_log (dict): A dictionary where the keys are variants (sequences of activities) and the values are the occurrences of each variant in the log.
    - activity_counts (dict): A dictionary containing the counts of each activity in the graph.
    - edge_counts (dict): A dictionary containing the counts of each edge in the graph.

    Returns:
    - float: The normalized ER value.
    
    """
    ER_sum = 0.0
    total_occurences = 0
    ER_list = []
    for variant, occurrence in variant_log.items():
        # Calculate the replay probability of the trace with the real dfg
        prob = get_probability(activity_counts, edge_counts, variant)
        # Get a new dfg, which is only discovered using the varint, used for normalization
        act_counts_var, edge_count_var = get_dfg({variant:1})
        # The probability of this dfg is the maximal probability possible for this trace when using dfg's, not always 1 because of loops
        maximal_prob = get_probability(act_counts_var, edge_count_var, variant)
        # Get normalized probability, subtracting the minimal ER at the end (obtained with maximal probability) would be the same
        prob_norm = prob/maximal_prob

        er_trace = -math.log(prob_norm, 2)
        ER_sum += er_trace * occurrence
        total_occurences += occurrence
        ER_list.extend([er_trace] * occurrence)

    ER = ER_sum / total_occurences

    variance = sum((er - ER) ** 2 for er in ER_list) / total_occurences
    ER_std = math.sqrt(variance)

    return ER, ER_std


def compute_er_full(cases:  List, normalized: bool=True):
    full_variant_log = defaultdict(int)

    for case in cases:
        full_trace = tuple(case[0, :, 0].astype(int).tolist())
        full_variant_log[full_trace] += 1

    act_full, edge_full = get_dfg(dict(full_variant_log))
    if normalized:
        return get_ER_normalized(dict(full_variant_log), act_full, edge_full)
    return get_ER(dict(full_variant_log), act_full, edge_full)


def compute_er_segments(segmented_cases:  List[List[List[float]]], normalized: bool=True):
    seg_variant_log = defaultdict(int)

    for case in segmented_cases:
        for seg in case:
            seg = tuple(map(int, seg))
            seg_variant_log[seg] += 1

    act, edge = get_dfg(dict(seg_variant_log))
    if normalized:
        return get_ER_normalized(dict(seg_variant_log), act, edge)
    return get_ER(dict(seg_variant_log), act, edge)


def compute_er_full_and_segments(sv_list, cases_list, normalized=True):
    """
    Compute Entropic Relevance for full traces and for extracted segments.

    Builds two variant logs from the provided data:
    - full-trace log : each trace as a tuple of activity codes
    - segment log    : each individual segment as a tuple of activity codes

    A DFG is mined from each log and the corresponding ER (or normalised ER)
    is computed.

    Parameters
    ----------
    sv_list : list of dict
        One dict per trace, each containing at least:
        - ``"segment_ids"`` : list of lists of event indices (one list per segment)
    cases_list : list of np.ndarray, shape (1, trace_len, n_features)
        Raw feature arrays; activity code is assumed to be feature index 0.
    normalized : bool, optional
        When True (default) use ``get_ER_normalized``; otherwise ``get_ER``.

    Returns
    -------
    er_full : float
        ER for the full-trace variant log.
    er_segments : float
        ER for the segment variant log.
    """
    full_variant_log = defaultdict(int)
    segment_variant_log = defaultdict(int)

    for i, sv_dict in enumerate(sv_list):
        case = cases_list[i]          # shape (1, trace_len, n_features)
        seg_ids = sv_dict["segment_ids"]
        trace_len = seg_ids[-1][-1] + 1

        full_trace = tuple(int(case[0, t, 0]) for t in range(trace_len))
        full_variant_log[full_trace] += 1

        for seg_indices in seg_ids:
            seg_variant = tuple(int(case[0, idx, 0]) for idx in seg_indices)
            segment_variant_log[seg_variant] += 1

    act_full, edge_full = get_dfg(dict(full_variant_log))
    act_seg, edge_seg = get_dfg(dict(segment_variant_log))

    if normalized:
        er_full = get_ER_normalized(dict(full_variant_log), act_full, edge_full)
        er_seg = get_ER_normalized(dict(segment_variant_log), act_seg, edge_seg)
    else:
        er_full = get_ER(dict(full_variant_log), act_full, edge_full)
        er_seg = get_ER(dict(segment_variant_log), act_seg, edge_seg)

    return er_full, er_seg