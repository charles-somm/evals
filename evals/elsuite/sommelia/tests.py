import json
import pytest

from validation import (
    parse_response,
    normalize_string,
    check_wines,
    validate_recommendation,
)
from validation import Recommendation


@pytest.fixture
def test_valid_recommendation():
    return {
        "wines": [
            {
                "name": "Saint Émilion - Chateau Grand Mouton",
                "color": "red",
                "grapes": ["Merlot"],
                "appellation": "Saint-Émilion",
                "country": "FR",
                "explanation": "Le Château Grand Mouton est un vin rouge classique de Saint-Émilion. Avec son profil fruité et ses tanins souples, il s'accordera à merveille avec le caractère épicé du chili con carne. C'est un choix sûr pour ceux qui cherchent un vin de qualité à un prix raisonnable.",
            },
            {
                "name": "Château de Marsannay Gevrey-Chambertin 2019",
                "color": "white",
                "grapes": ["Chardonnay"],
                "appellation": "Gevrey-Chambertin",
                "country": "FR",
                "explanation": "Bien que Gevrey-Chambertin soit plus connu pour ses vins rouges, ce Chardonnay offre une alternative élégante et pleine de caractère. Avec sa structure et sa richesse, il complétera la blanquette de veau et la tarte aux poireaux, tout en étant suffisamment souple pour s'accorder avec différents fromages. C'est un choix distingué pour un cadeau qui reste dans une gamme de prix raisonnable.",
            },
        ]
    }


@pytest.fixture
def test_invalid_recommendation():
    return {
        "wines": [
            {
                "name": "Château Grand Mouton",
                "color": "red",
                "grapes": ["Merlot"],
                "appellation": "Saint-Émilion",
                "country": "FR",
                "explanation": "Le Château Grand Mouton est un vin rouge classique de Saint-Émilion. Avec son profil fruité et ses tanins souples, il s'accordera à merveille avec le caractère épicé du chili con carne. C'est un choix sûr pour ceux qui cherchent un vin de qualité à un prix raisonnable.",
            },
            {
                "name": "Domaine de la Romanée-Conti Romanée-Conti 2019",
                "color": "rosé",
                "grapes": ["Pinot Noir"],
                "appellation": "Romanée-Conti",
                "country": "FR",
                "explanation": "",
            },
        ]
    }


@pytest.fixture
def test_listed_wines():
    return [
        "Château Grand Mouton",
        "Château de Marsannay Gevrey-Chambertin 2019",
        "Domaine de la Romanée-Conti Romanée-Conti 2019",
    ]


def test_normalize_string():
    test_string = "Château d'Yquem 2019"
    test_string2 = "ceci ne change pas"

    assert normalize_string(test_string) == "chateau d yquem 2019"
    assert normalize_string(test_string2) == "ceci ne change pas"


def test_parse_response():
    test_sample = {
        "wines": [
            {
                "name": "Château Grand Mouton",
                "color": "red",
                "grapes": ["Merlot"],
                "appellation": "Saint-Émilion",
                "country": "FR",
                "explanation": "Le Château Grand Mouton est un vin rouge classique de Saint-Émilion. Avec son profil fruité et ses tanins souples, il s'accordera à merveille avec le caractère épicé du chili con carne. C'est un choix sûr pour ceux qui cherchent un vin de qualité à un prix raisonnable.",
            },
            {
                "name": "Château de Marsannay Gevrey-Chambertin 2019",
                "color": "white",
                "grapes": ["Chardonnay"],
                "appellation": "Gevrey-Chambertin",
                "country": "FR",
                "explanation": "Bien que Gevrey-Chambertin soit plus connu pour ses vins rouges, ce Chardonnay offre une alternative élégante et pleine de caractère. Avec sa structure et sa richesse, il complétera la blanquette de veau et la tarte aux poireaux, tout en étant suffisamment souple pour s'accorder avec différents fromages. C'est un choix distingué pour un cadeau qui reste dans une gamme de prix raisonnable.",
            },
        ]
    }
    test_json = json.dumps(test_sample)
    assert parse_response(test_json) == Recommendation(**test_sample)


def test_check_wines():
    test_listed_wines = [
        "Château Grand Mouton",
        "Château de Marsannay Gevrey-Chambertin 2019",
        "Domaine de la Romanée-Conti Romanée-Conti 2019",
    ]
    test_valid_recommended_wines = ["Grand Mouton", "Chateau Gevrey-Chambertin"]
    test_invalid_recommended_wines = ["Grand Mouton", "Marsannay Conti"]

    assert check_wines(test_valid_recommended_wines, test_listed_wines) == True
    assert check_wines(test_invalid_recommended_wines, test_listed_wines) == False


def test_validate_recommendation(
    test_valid_recommendation, test_invalid_recommendation, test_listed_wines
):
    valid_recommendation_json = json.dumps(test_valid_recommendation)
    invalid_recommendation_json = json.dumps(test_invalid_recommendation)

    assert validate_recommendation(valid_recommendation_json, test_listed_wines) == {
        # "recommendation": Recommendation(**test_valid_recommendation).model_dump(),
        "is_valid": True,
        "error": None,
        "are_wines_in_list": True,
    }

    assert validate_recommendation(invalid_recommendation_json, test_listed_wines) == {
        # "recommendation": invalid_recommendation_json,
        "is_valid": False,
        "error": "('wines', 1, 'explanation'): Value error, Explanation is empty.",
        "are_wines_in_list": False,
    }
