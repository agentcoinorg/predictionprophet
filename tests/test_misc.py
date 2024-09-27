from prediction_prophet.autonolas.research import (
    fields_dict_to_bullet_list,
    list_to_list_str,
)


def test_list_to_list_str() -> None:
    list_to_list_str(["foo", "bar", "baz"]) == '"foo", "bar" and "baz"'


def test_fields_dict_to_bullet_list() -> None:
    field_dict = {"foo": "foo sth", "bar": "bar sth", "baz": "baz sth"}
    EXPECTED_LIST_STR = """  - foo: foo sth
  - bar: bar sth
  - baz: baz sth"""
    assert fields_dict_to_bullet_list(field_dict) == EXPECTED_LIST_STR
