"""Strict validation for the display-only A2UI v1.0 message subset."""

import json
import math
import re
from collections.abc import Mapping
from typing import Any

A2UI_BASIC_CATALOG_ID = (
    "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"
)

MAX_BATCH_MESSAGES = 50
MAX_COMPONENTS_PER_MESSAGE = 100
MAX_SURFACE_ID_LENGTH = 128
MAX_COMPONENT_ID_LENGTH = 128
MAX_STRING_LENGTH = 10_000
MAX_BATCH_BYTES = 256_000
MAX_DATA_DEPTH = 12
MAX_COMPONENT_WEIGHT = 1_000

ALLOWED_OPERATIONS = {
    "createSurface",
    "updateComponents",
    "updateDataModel",
    "deleteSurface",
}

ALLOWED_COMPONENTS = {
    "Text",
    "Row",
    "Column",
    "List",
    "Card",
    "Divider",
}

COMPONENT_PROPERTIES = {
    "Text": {"text", "variant", "weight"},
    "Row": {"children", "justify", "align", "weight"},
    "Column": {"children", "justify", "align", "weight"},
    "List": {"children", "direction", "align", "weight"},
    "Card": {"child", "weight"},
    "Divider": {"axis", "weight"},
}

JUSTIFY_VALUES = {
    "center",
    "end",
    "spaceAround",
    "spaceBetween",
    "spaceEvenly",
    "start",
    "stretch",
}

ALIGN_VALUES = {"start", "center", "end", "stretch"}

RESERVED_KEYS = {
    "__proto__",
    "prototype",
    "constructor",
    "@call",
    "call",
}


def _require_record(value: Any, field: str) -> Mapping[str, Any]:
    """Require a JSON object."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    field: str,
) -> None:
    """Reject missing required fields and unexpected fields."""
    keys = set(value)

    missing = required - keys
    unexpected = keys - required - optional

    if missing:
        raise ValueError(f"{field} is missing required fields: {sorted(missing)}")

    if unexpected:
        raise ValueError(f"{field} contains unsupported fields: {sorted(unexpected)}")


def _validate_identifier(value: Any, field: str) -> None:
    """Validate a bounded, non-empty protocol identifier."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_COMPONENT_ID_LENGTH
        or not value.strip()
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{field} must be a valid non-empty identifier.")


def _validate_component_identifier(value: Any, field: str) -> None:
    """Validate a component ID using Unicode identifier rules (UAX #31)."""
    _validate_identifier(value, field)

    if not value.isidentifier():
        raise ValueError(f"{field} must be a valid UAX #31 identifier.")


def _validate_string(value: Any, field: str) -> None:
    """Validate bounded text without control characters."""
    if (
        not isinstance(value, str)
        or len(value) > MAX_STRING_LENGTH
        or any(ord(char) < 32 and char not in "\n\t\r" for char in value)
    ):
        raise ValueError(f"{field} must be a valid bounded string.")


def _validate_data_binding(value: Any, field: str) -> bool:
    """Accept only a single-path A2UI data binding, never a function call."""
    if not isinstance(value, Mapping):
        return False

    if set(value) not in ({"@path"}, {"path"}):
        return False

    path = next(iter(value.values()))

    if (
        not isinstance(path, str)
        or not path.startswith("/")
        or len(path) > 512
        or "\x00" in path
    ):
        raise ValueError(f"{field} contains an invalid data binding path.")

    return True


def _validate_json_value(
    value: Any,
    field: str,
    depth: int = 0,
) -> None:
    """Validate bounded JSON data and reject executable expressions."""
    if depth > MAX_DATA_DEPTH:
        raise ValueError(f"{field} exceeds the maximum nesting depth.")

    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str):
            _validate_string(value, field)
        return

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number.")
        return

    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field}[{index}]", depth + 1)
        return

    if isinstance(value, Mapping):
        if _validate_data_binding(value, field):
            return

        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string key.")

            if key in RESERVED_KEYS or key in {"@path", "path"}:
                raise ValueError(f"{field} contains a reserved expression key.")

            _validate_string(key, f"{field} key")
            _validate_json_value(item, f"{field}.{key}", depth + 1)
        return

    raise ValueError(f"{field} contains an unsupported JSON value.")


def _validate_weight(component: Mapping[str, Any], field: str) -> None:
    """Validate the Basic Catalog weight property."""
    if "weight" not in component:
        return

    weight = component["weight"]

    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight < 0
        or weight > MAX_COMPONENT_WEIGHT
    ):
        raise ValueError(f"{field}.weight must be between 0 and 1000.")


def _validate_child_ids(value: Any, field: str) -> None:
    """Accept static child references; reject dynamic templates."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list of component IDs.")

    for index, child_id in enumerate(value):
        _validate_component_identifier(child_id, f"{field}[{index}]")


def _validate_component(component: Any, index: int) -> None:
    """Validate one component against the restricted Basic Catalog subset."""
    field = f"components[{index}]"
    component = _require_record(component, field)

    if "id" not in component or "component" not in component:
        raise ValueError(f"{field} requires id and component.")

    component_id = component["id"]
    component_type = component["component"]

    _validate_component_identifier(component_id, f"{field}.id")

    if not isinstance(component_type, str) or component_type not in ALLOWED_COMPONENTS:
        raise ValueError(
            f"{field} uses unsupported component type: {component_type!r}."
        )

    allowed = {"id", "component", "@catalogId"} | COMPONENT_PROPERTIES[component_type]

    unexpected = set(component) - allowed
    if unexpected:
        raise ValueError(
            f"{field} contains unsupported properties: {sorted(unexpected)}"
        )

    if "@catalogId" in component:
        if component["@catalogId"] != A2UI_BASIC_CATALOG_ID:
            raise ValueError(f"{field} specifies an unsupported catalog.")

    _validate_weight(component, field)

    if component_type == "Text":
        if "text" not in component:
            raise ValueError(f"{field}.text is required.")

        text_value = component["text"]

        if isinstance(text_value, str):
            _validate_string(text_value, f"{field}.text")

            # Basic Catalog Text supports simple Markdown, not HTML,
            # Markdown links, or embedded images.
            if (
                re.search(r"<[^>]*>", text_value)
                or "![" in text_value
                or re.search(r"\]\(", text_value)
            ):
                raise ValueError(f"{field}.text contains unsupported markup.")
        elif not _validate_data_binding(text_value, f"{field}.text"):
            raise ValueError(f"{field}.text must be plain text or a data binding.")

        variant = component.get("variant", "body")
        if not isinstance(variant, str) or variant not in {"body", "caption"}:
            raise ValueError(f"{field}.variant is invalid.")

    elif component_type in {"Row", "Column", "List"}:
        if "children" not in component:
            raise ValueError(f"{field}.children is required.")

        _validate_child_ids(component["children"], f"{field}.children")

        if component_type in {"Row", "Column"}:
            justify = component.get("justify", "start")
            align = component.get("align", "stretch")

            if not isinstance(justify, str) or justify not in JUSTIFY_VALUES:
                raise ValueError(f"{field}.justify is invalid.")

            if not isinstance(align, str) or align not in ALIGN_VALUES:
                raise ValueError(f"{field}.align is invalid.")

        if component_type == "List":
            direction = component.get("direction", "vertical")
            align = component.get("align", "stretch")

            if not isinstance(direction, str) or direction not in {
                "vertical",
                "horizontal",
            }:
                raise ValueError(f"{field}.direction is invalid.")

            if not isinstance(align, str) or align not in ALIGN_VALUES:
                raise ValueError(f"{field}.align is invalid.")

    elif component_type == "Card":
        if "child" not in component:
            raise ValueError(f"{field}.child is required.")

        _validate_component_identifier(component["child"], f"{field}.child")
    elif component_type == "Divider":
        axis = component.get("axis", "horizontal")
        if not isinstance(axis, str) or axis not in {"horizontal", "vertical"}:
            raise ValueError(f"{field}.axis is invalid.")


def _validate_components(value: Any, field: str) -> None:
    """Validate a component collection and reject duplicate IDs."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty component list.")

    if len(value) > MAX_COMPONENTS_PER_MESSAGE:
        raise ValueError(f"{field} exceeds the component limit.")

    identifiers: set[str] = set()

    for index, component in enumerate(value):
        _validate_component(component, index)
        component_id = component["id"]

        if component_id in identifiers:
            raise ValueError(f"{field} contains duplicate component IDs.")

        identifiers.add(component_id)


def _validate_message(message: Any, index: int) -> None:
    """Validate one supported A2UI v1.0 protocol message."""
    field = f"messages[{index}]"
    message = _require_record(message, field)

    if message.get("version") != "1.0":
        raise ValueError(f"{field}.version must be '1.0'.")

    operations = set(message) - {"version"}

    if len(operations) != 1:
        raise ValueError(f"{field} must contain exactly one operation.")

    operation = next(iter(operations))

    if operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"{field} uses an unsupported operation.")

    payload = _require_record(message[operation], f"{field}.{operation}")

    if operation == "createSurface":
        _require_exact_keys(
            payload,
            {"surfaceId"},
            {"catalogId", "sendDataModel", "components", "dataModel"},
            f"{field}.createSurface",
        )

        _validate_identifier(
            payload["surfaceId"],
            f"{field}.createSurface.surfaceId",
        )

        if "catalogId" in payload:
            if payload["catalogId"] != A2UI_BASIC_CATALOG_ID:
                raise ValueError(f"{field} specifies an unsupported catalog.")

        if payload.get("sendDataModel", False) is not False:
            raise ValueError(
                f"{field} cannot enable sendDataModel in display-only mode."
            )

        if "components" in payload:
            _validate_components(
                payload["components"],
                f"{field}.createSurface.components",
            )

        if "dataModel" in payload:
            _validate_json_value(
                payload["dataModel"],
                f"{field}.createSurface.dataModel",
            )
            _require_record(
                payload["dataModel"],
                f"{field}.createSurface.dataModel",
            )

    elif operation == "updateComponents":
        _require_exact_keys(
            payload,
            {"surfaceId", "components"},
            set(),
            f"{field}.updateComponents",
        )

        _validate_identifier(
            payload["surfaceId"],
            f"{field}.updateComponents.surfaceId",
        )
        _validate_components(
            payload["components"],
            f"{field}.updateComponents.components",
        )

    elif operation == "updateDataModel":
        _require_exact_keys(
            payload,
            {"surfaceId", "value"},
            {"path"},
            f"{field}.updateDataModel",
        )

        _validate_identifier(
            payload["surfaceId"],
            f"{field}.updateDataModel.surfaceId",
        )

        if "path" in payload:
            path = payload["path"]
            if (
                not isinstance(path, str)
                or not path.startswith("/")
                or len(path) > 512
                or "\x00" in path
            ):
                raise ValueError(f"{field}.updateDataModel.path is invalid.")

        _validate_json_value(
            payload["value"],
            f"{field}.updateDataModel.value",
        )

    elif operation == "deleteSurface":
        _require_exact_keys(
            payload,
            {"surfaceId"},
            set(),
            f"{field}.deleteSurface",
        )

        _validate_identifier(
            payload["surfaceId"],
            f"{field}.deleteSurface.surfaceId",
        )


def validate_a2ui_message_batch(payload: Any) -> list[dict[str, Any]]:
    """Validate a bounded batch of safe A2UI v1.0 display messages."""
    if not isinstance(payload, list) or not payload:
        raise ValueError("A2UI payload must be a non-empty message list.")

    if len(payload) > MAX_BATCH_MESSAGES:
        raise ValueError("A2UI message batch exceeds the message limit.")

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("A2UI payload must contain valid JSON data.") from exc

    if len(encoded) > MAX_BATCH_BYTES:
        raise ValueError("A2UI message batch exceeds the size limit.")

    for index, message in enumerate(payload):
        _validate_message(message, index)

    return payload
