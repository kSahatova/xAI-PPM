import string
from jellyfish import damerau_levenshtein_distance


def control_flow_log_distance_simplified(
    trace1, trace2, total_events_num=25, special_tokens_num=3
):
    """
    Compute the Control-Flow Log Distance (see "Camargo M, Dumas M, González-Rojas O. 2021. Discovering generative models
    from event logs: data-driven simulation vs deep learning. PeerJ Computer Science 7:e577 https://doi.org/10.7717/peerj-cs.577"
    for a detailed description of a similarity version of the metric).

    :param original_log: first event log.
    :param original_ids: mapping for the column IDs of the first event log.

    :return: the Control-Flow Log Distance measure between [trace_1] and [trace_2].
    """
    # Transform the event log to a list of character sequences representing the traces
    characters = string.ascii_uppercase
    start_ind = special_tokens_num
    end_ind = start_ind + total_events_num
    event_to_symbol_mapping = {
        i: characters[i - special_tokens_num] for i in range(start_ind, end_ind)
    }
    trace1_cat = trace1[0].squeeze().numpy()
    trace2_cat = trace2[0].squeeze().numpy()
    activity_sequence1 = "".join(
        [event_to_symbol_mapping[activity] for activity in trace1_cat]
    )
    activity_sequence2 = "".join(
        [event_to_symbol_mapping[activity] for activity in trace2_cat]
    )

    # Calculate the DL distance between each pair of traces
    distance = damerau_levenshtein_distance(activity_sequence1, activity_sequence2)
    return distance
