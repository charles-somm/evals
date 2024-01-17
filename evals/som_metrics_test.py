from evals.record import Event
from evals.som_metrics import get_accuracy


def test_get_accuracy():
    events = [
        Event(
            type="validation",
            data={"is_valid": True, "are_wines_in_list": True},
            sample_id=0,
            run_id=0,
            event_id=0,
            created_by="",
            created_at=0,
        ),
        Event(
            type="validation",
            data={"is_valid": True, "are_wines_in_list": False},
            sample_id=1,
            run_id=0,
            event_id=1,
            created_by="",
            created_at=0,
        ),
        Event(
            type="validation",
            data={"is_valid": False, "are_wines_in_list": True},
            sample_id=2,
            run_id=0,
            event_id=2,
            created_by="",
            created_at=0,
        ),
        Event(
            type="validation",
            data={"is_valid": False, "are_wines_in_list": False},
            sample_id=3,
            run_id=0,
            event_id=3,
            created_by="",
            created_at=0,
        ),
    ]
    assert get_accuracy(events) == {
        "valid_model": 0.5,
        "valid_wines": 0.5,
    }
