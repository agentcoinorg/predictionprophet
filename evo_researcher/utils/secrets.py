import os
from pydantic.types import SecretStr


def secret_str_from_env(key: str) -> SecretStr | None:
    value = os.getenv(key)
    return SecretStr(value) if value else None
