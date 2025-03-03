from prediction_prophet.autonolas.research import (
    fields_dict_to_bullet_list,
    list_to_list_str,
)
import pytest
import json
from prediction_prophet.autonolas.research import clean_completion_json


def test_list_to_list_str() -> None:
    list_to_list_str(["foo", "bar", "baz"]) == '"foo", "bar" and "baz"'


def test_fields_dict_to_bullet_list() -> None:
    field_dict = {"foo": "foo sth", "bar": "bar sth", "baz": "baz sth"}
    EXPECTED_LIST_STR = """  - foo: foo sth
  - bar: bar sth
  - baz: baz sth"""
    assert fields_dict_to_bullet_list(field_dict) == EXPECTED_LIST_STR

@pytest.mark.parametrize("input, expected", [
    ("""{
        "foo": "foo sth"
    }""", {"foo": "foo sth"}),
    ("""{
        "foo": "foo 
    sth"
    }""", {"foo": "foo sth"}),
])
def test_clean_completion_json(input: str, expected: str) -> None:
    clean = clean_completion_json(input)
    assert json.loads(clean) == expected, clean
