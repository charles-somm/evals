from numpy import nan
from typing import Optional, Sequence, Set

from evals.record import Event


def get_accuracy(events: Sequence[Event]) -> dict[str, float]:
    """Aggregates multiple metrics included in the event.
    - is_valid: whether the response complies with the model
    - are_wines_in_list: whether the wines in the response are in the list of wines
    """
    num_valid_model = sum(int(event.data["is_valid"]) for event in events)
    num_valid_wines = sum(int(event.data["are_wines_in_list"]) for event in events)
    num_total = len(events)
    if num_total == 0:
        return {
            "valid_model": nan,
            "valid_wines": nan,
        }
    else:
        return {
            "valid_model": num_valid_model / num_total,
            "valid_wines": num_valid_wines / num_total,
        }
