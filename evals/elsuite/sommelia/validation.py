import re
import unicodedata

from thefuzz import fuzz
from thefuzz import process
from pydantic import BaseModel, Field, ValidationError, field_validator
from enum import Enum


class WineColor(Enum):
    RED = "red"
    WHITE = "white"
    ROSE = "rose"


class Wine(BaseModel):
    name: str
    color: WineColor = Field(..., description="Color of the wine.")
    appellation: str
    country: str
    grapes: list[str]
    explanation: str

    @field_validator("explanation")
    def validate_explanation(cls, explanation):
        if not explanation.strip():
            raise ValueError("Explanation is empty.")
        else:
            return explanation

    @field_validator("name")
    def validate_name(cls, name):
        if not name.strip():
            raise ValueError("Name is empty.")
        else:
            return name


class Recommendation(BaseModel):
    wines: list[Wine]

    @field_validator("wines")
    def validate_wines(cls, wines):
        if len(wines) != 2:
            raise ValueError(f"Must provide 2 wines. {len(wines)} provided.")
        else:
            return wines

    # TODO: recommendation can be empty if recommendation is impossible


def parse_response(sampled: str):
    return Recommendation.model_validate_json(sampled)


def normalize_string(name: str):
    """Normalize a string to lowercase and remove accents and non alpha-numeric
    characters for comparison.
    """
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("utf-8")
    # Remove non-alphanumeric characters
    name = re.sub(r"\W+", " ", name)
    return name.lower()


def check_wines(recommended_wines: list[str], listed_wines: list[str]):
    """Check if the wines in the recommendation are in the list of wines."""

    for wine in recommended_wines:
        if not any(
            fuzz.partial_token_sort_ratio(
                normalize_string(wine), normalize_string(listed_wine)
            )
            > 80
            for listed_wine in listed_wines
        ):
            return False
    return True


def validate_recommendation(sampled: str, listed_wines: list[str]):
    """Validate a recommendation."""

    recommendation = sampled
    error = None
    is_valid = False
    are_wines_in_list = False

    # Check model
    try:
        recommendation = parse_response(sampled).model_dump()
        is_valid = True
    except ValidationError as e:
        print(f"ERROR: {repr(e.errors())}")
        is_valid = False
        error_loc = e.errors()[0].get("loc")
        error_msg = e.errors()[0].get("msg")
        error = f"{error_loc}: {error_msg}"

    # Check if the wines in the recommendation are in the list of wines
    if is_valid:
        are_wines_in_list = check_wines(
            [wine["name"] for wine in recommendation["wines"]], listed_wines
        )

    return {
        "is_valid": is_valid,
        "error": error,
        "are_wines_in_list": are_wines_in_list,
    }
