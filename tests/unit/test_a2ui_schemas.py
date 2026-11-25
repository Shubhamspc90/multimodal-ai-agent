"""Unit tests for strict display-only A2UI message validation."""

import pytest

from backend.app.schemas.a2ui import (
    A2UI_BASIC_CATALOG_ID,
    validate_a2ui_message_batch,
)


def test_accepts_valid_display_only_message_batch() -> None:
    """Accept a valid Basic Catalog surface with static components."""
    messages = [
        {
            "version": "1.0",
            "createSurface": {
                "surfaceId": "surface-1",
                "catalogId": A2UI_BASIC_CATALOG_ID,
                "sendDataModel": False,
            },
        },
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "root",
                        "component": "Column",
                        "children": ["heading", "body"],
                    },
                    {
                        "id": "heading",
                        "component": "Text",
                        "text": "Welcome",
                        "variant": "body",
                    },
                    {
                        "id": "body",
                        "component": "Text",
                        "text": "A2UI is ready.",
                    },
                ],
            },
        },
    ]

    assert validate_a2ui_message_batch(messages) == messages


def test_accepts_valid_data_binding() -> None:
    """Allow a simple JSON Pointer binding without function execution."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "heading",
                        "component": "Text",
                        "text": {"@path": "/title"},
                    }
                ],
            },
        }
    ]

    assert validate_a2ui_message_batch(messages) == messages


def test_accepts_supported_layout_properties() -> None:
    """Accept valid layout and divider properties."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "row",
                        "component": "Row",
                        "children": ["item"],
                        "justify": "spaceBetween",
                        "align": "center",
                    },
                    {
                        "id": "item",
                        "component": "Text",
                        "text": "Item",
                    },
                    {
                        "id": "divider",
                        "component": "Divider",
                        "axis": "horizontal",
                    },
                ],
            },
        }
    ]

    assert validate_a2ui_message_batch(messages) == messages


def test_accepts_card_with_child_reference() -> None:
    """Accept a card referencing a component by ID."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "card",
                        "component": "Card",
                        "child": "cardContent",
                    },
                    {
                        "id": "cardContent",
                        "component": "Text",
                        "text": "Card content",
                    },
                ],
            },
        }
    ]

    assert validate_a2ui_message_batch(messages) == messages


@pytest.mark.parametrize(
    "component_id",
    [
        "details-card",
        "1details",
        "details card",
        "",
    ],
)
def test_rejects_invalid_component_identifiers(component_id: str) -> None:
    """Reject component IDs that violate UAX #31 identifier rules."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": component_id,
                        "component": "Text",
                        "text": "Student details",
                    }
                ],
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


def test_rejects_invalid_child_component_identifier() -> None:
    """Reject invalid IDs in static child references."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "root",
                        "component": "Column",
                        "children": ["details-card"],
                    }
                ],
            },
        }
    ]

    with pytest.raises(ValueError, match="UAX #31"):
        validate_a2ui_message_batch(messages)


def test_accepts_camel_case_component_identifiers() -> None:
    """Accept valid camelCase component IDs and child references."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "studentDetails",
                        "component": "Column",
                        "children": ["studentName"],
                    },
                    {
                        "id": "studentName",
                        "component": "Text",
                        "text": "Rahul Sharma",
                    },
                ],
            },
        }
    ]

    assert validate_a2ui_message_batch(messages) == messages


@pytest.mark.parametrize(
    "messages",
    [
        [],
        {},
        None,
        [{"version": "0.9", "deleteSurface": {"surfaceId": "s1"}}],
        [{"version": "1.0", "callRendererFunction": {}}],
        [{"version": "1.0", "agentFunctionResponse": {}}],
        [
            {
                "version": "1.0",
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://attacker.example/catalog.json",
                },
            }
        ],
        [
            {
                "version": "1.0",
                "createSurface": {
                    "surfaceId": "s1",
                    "sendDataModel": True,
                },
            }
        ],
        [
            {
                "version": "1.0",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "button",
                            "component": "Button",
                            "child": "text",
                        }
                    ],
                },
            }
        ],
        [
            {
                "version": "1.0",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "text",
                            "component": "Text",
                            "text": "Hello",
                            "action": {
                                "event": {"name": "execute"},
                            },
                        }
                    ],
                },
            }
        ],
        [
            {
                "version": "1.0",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "text",
                            "component": "Text",
                            "text": {
                                "@call": "openUrl",
                                "args": {"url": "https://example.com"},
                            },
                        }
                    ],
                },
            }
        ],
        [
            {
                "version": "1.0",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "text",
                            "component": "Text",
                            "text": "Hello",
                            "unexpected": "value",
                        }
                    ],
                },
            }
        ],
        [
            {
                "version": "1.0",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "duplicate",
                            "component": "Text",
                            "text": "One",
                        },
                        {
                            "id": "duplicate",
                            "component": "Text",
                            "text": "Two",
                        },
                    ],
                },
            }
        ],
        [
            {
                "version": "1.0",
                "updateDataModel": {
                    "surfaceId": "s1",
                    "value": {
                        "action": {
                            "@call": "openUrl",
                            "args": {"url": "https://example.com"},
                        }
                    },
                },
            }
        ],
    ],
)
def test_rejects_unsupported_or_unsafe_messages(messages: object) -> None:
    """Reject malformed, interactive, or unsupported protocol messages."""
    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


def test_rejects_html_and_markdown_links_in_text() -> None:
    """Keep Text content within the Basic Catalog's safe Markdown subset."""
    for content in (
        "<script>alert(1)</script>",
        "[Open website](https://example.com)",
        "![image](https://example.com/image.png)",
    ):
        messages = [
            {
                "version": "1.0",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "text",
                            "component": "Text",
                            "text": content,
                        }
                    ],
                },
            }
        ]

        with pytest.raises(ValueError):
            validate_a2ui_message_batch(messages)


def test_rejects_non_finite_numbers_in_data_model() -> None:
    """Reject NaN and infinity from the JSON data model."""
    messages = [
        {
            "version": "1.0",
            "updateDataModel": {
                "surfaceId": "s1",
                "value": {"score": float("nan")},
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


def test_rejects_dynamic_child_templates() -> None:
    """Reject data-driven component templates in the initial safe subset."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "s1",
                "components": [
                    {
                        "id": "list",
                        "component": "List",
                        "children": {
                            "componentId": "item",
                            "path": "/items",
                        },
                    }
                ],
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


def test_rejects_oversized_batch() -> None:
    """Enforce the message-count limit."""
    messages = [
        {
            "version": "1.0",
            "deleteSurface": {"surfaceId": "s1"},
        }
        for _ in range(51)
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


@pytest.mark.parametrize(
    "component_type",
    [
        {},
        [],
        None,
        123,
    ],
)
def test_rejects_invalid_component_type(component_type: object) -> None:
    """Reject non-string component types without leaking TypeError."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "component1",
                        "component": component_type,
                        "text": "Hello",
                    }
                ],
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


def test_rejects_invalid_text_variant_type() -> None:
    """Reject malformed enum values with a validation error."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "heading",
                        "component": "Text",
                        "text": "Welcome",
                        "variant": [],
                    }
                ],
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


def test_rejects_invalid_card_child_identifier() -> None:
    """Reject invalid component IDs referenced by a Card."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "surface-1",
                "components": [
                    {
                        "id": "studentCard",
                        "component": "Card",
                        "child": "card-content",
                    },
                    {
                        "id": "cardContent",
                        "component": "Text",
                        "text": "Student details",
                    },
                ],
            },
        }
    ]

    with pytest.raises(ValueError, match="UAX #31"):
        validate_a2ui_message_batch(messages)
