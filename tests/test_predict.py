from src.predict import suggest_priority


def test_high_priority_for_compromised_card():
    result = suggest_priority(
        "My card was used by someone else",
        "compromised_card",
    )
    assert result == "HIGH"


def test_high_priority_for_urgent_ticket():
    result = suggest_priority(
        "This is urgent, please help me",
        "card_arrival",
    )
    assert result == "HIGH"


def test_normal_priority_for_regular_ticket():
    result = suggest_priority(
        "When will my new card arrive?",
        "card_arrival",
    )
    assert result == "NORMAL"