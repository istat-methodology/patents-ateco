from typing import Any


INVALID_CODE = "INVALID"


def normalize_code(code: Any) -> str:
    """
    Normalizza un codice NACE in formato stringa corretto.
    Gestisce casi tipo:
    - 62.1 -> 62.10
    - ' 62.10 ' -> 62.10
    - None -> ""
    """
    if code is None:
        return ""

    code = str(code).strip()

    if not code:
        return ""

    # correzione float-like: 62.1 -> 62.10
    if "." in code:
        left, right = code.split(".", 1)

        if right.isdigit() and len(right) == 1:
            code = f"{left}.{right}0"

    return code


def normalize_secondary_codes(value: Any) -> list[str]:
    """
    Converte secondary_codes in lista di stringhe pulite.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [normalize_code(x) for x in value if normalize_code(x)]

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return []

        # fallback semplice
        return [normalize_code(value)]

    return []


def deduplicate_preserve_order(values: list[str]) -> list[str]:
    """
    Rimuove duplicati mantenendo ordine.
    """
    seen = set()
    result = []

    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)

    return result


def validate_classification_result(
    result: dict,
    valid_codes: set[str],
    max_secondary: int = 3,
) -> dict:
    """
    Valida e pulisce il risultato prodotto dal modello.

    Output garantito:
    {
        "primary_code": "...",
        "secondary_codes": ["...", "..."]
    }
    """

    if not isinstance(result, dict):
        return {
            "primary_code": INVALID_CODE,
            "secondary_codes": [],
        }

    primary_code = normalize_code(result.get("primary_code"))
    secondary_codes = normalize_secondary_codes(result.get("secondary_codes"))

    # validazione primary
    if primary_code not in valid_codes:
        primary_code = INVALID_CODE

    # filtra secondary non validi
    secondary_codes = [code for code in secondary_codes if code in valid_codes]

    # rimuove duplicati
    secondary_codes = deduplicate_preserve_order(secondary_codes)

    # rimuove primary dai secondary
    secondary_codes = [code for code in secondary_codes if code != primary_code]

    # tronca lunghezza
    secondary_codes = secondary_codes[:max_secondary]

    return {
        "primary_code": primary_code,
        "secondary_codes": secondary_codes,
    }
