from sportsmodel.serving.team_names import match_team_names, norm_team


def test_norm_team_strips_accents_punct_case():
    assert norm_team("San José State Spartans") == "san jose state spartans"
    assert norm_team("Texas A&M Aggies") == "texas am aggies"


def test_match_team_names_only_exact_normalized_matches():
    m = match_team_names(["Buffalo Bills", "Hawai'i Rainbow Warriors", "Nowhere U"],
                         ["Buffalo Bills", "Hawaii Rainbow Warriors"])
    assert m == {"Buffalo Bills": "Buffalo Bills", "Hawai'i Rainbow Warriors": "Hawaii Rainbow Warriors"}
