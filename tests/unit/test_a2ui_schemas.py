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


def test_accepts_safe_interactive_form_components() -> None:
    """Accept bound inputs and a restricted form submission event."""
    messages = [
        {
            "version": "1.0",
            "createSurface": {
                "surfaceId": "formSurface",
                "catalogId": A2UI_BASIC_CATALOG_ID,
                "sendDataModel": False,
                "dataModel": {
                    "name": "",
                    "enrolled": False,
                    "subjects": [],
                    "score": 50,
                    "date": "",
                },
            },
        },
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "formSurface",
                "components": [
                    {
                        "id": "root",
                        "component": "Column",
                        "children": [
                            "nameField",
                            "enrolledField",
                            "subjectPicker",
                            "scoreSlider",
                            "dateField",
                            "submitLabel",
                            "submitButton",
                        ],
                    },
                    {
                        "id": "nameField",
                        "component": "TextField",
                        "label": "Student name",
                        "value": {"@path": "/name"},
                        "placeholder": "Enter name",
                    },
                    {
                        "id": "enrolledField",
                        "component": "CheckBox",
                        "label": "Currently enrolled",
                        "value": {"@path": "/enrolled"},
                    },
                    {
                        "id": "subjectPicker",
                        "component": "ChoicePicker",
                        "options": [
                            {"value": "math", "label": "Mathematics"},
                            {"value": "science", "label": "Science"},
                        ],
                        "value": {"@path": "/subjects"},
                        "variant": "multipleSelection",
                        "displayStyle": "checkbox",
                    },
                    {
                        "id": "scoreSlider",
                        "component": "Slider",
                        "min": 0,
                        "max": 100,
                        "value": {"@path": "/score"},
                        "steps": 1,
                    },
                    {
                        "id": "dateField",
                        "component": "DateTimeInput",
                        "value": {"@path": "/date"},
                        "enableDate": True,
                        "enableTime": False,
                    },
                    {
                        "id": "submitLabel",
                        "component": "Text",
                        "text": "Submit form",
                    },
                    {
                        "id": "submitButton",
                        "component": "Button",
                        "child": "submitLabel",
                        "action": {
                            "event": {"name": "submit_form"},
                        },
                        "variant": "primary",
                    },
                ],
            },
        },
    ]

    assert validate_a2ui_message_batch(messages) == messages


@pytest.mark.parametrize(
    "component",
    [
        {
            "id": "field",
            "component": "TextField",
            "label": "Name",
            "value": "static value",
        },
        {
            "id": "field",
            "component": "TextField",
            "label": "Name",
            "value": {"path": "/name"},
        },
        {
            "id": "field",
            "component": "CheckBox",
            "label": "Accept",
            "value": True,
        },
        {
            "id": "picker",
            "component": "ChoicePicker",
            "options": [{"value": "a"}],
            "value": {"@path": "/choice"},
            "variant": "unsupported",
        },
        {
            "id": "slider",
            "component": "Slider",
            "max": 100,
            "value": {"@call": "execute"},
        },
        {
            "id": "date",
            "component": "DateTimeInput",
            "value": {"@path": "/date"},
            "enableDate": "yes",
        },
        {
            "id": "button",
            "component": "Button",
            "child": "label",
            "action": {
                "functionCall": {
                    "name": "execute",
                },
            },
        },
        {
            "id": "button",
            "component": "Button",
            "child": "label",
            "action": {
                "event": {"name": "open_url"},
            },
        },
        {
            "id": "button",
            "component": "Button",
            "child": "label",
            "action": {
                "event": {
                    "name": "submit_form",
                    "context": {"url": "https://attacker.example"},
                },
            },
        },
        {
            "id": "button",
            "component": "Button",
            "child": "label",
            "action": {
                "event": {"name": "submit_form"},
            },
            "@checks": [],
        },
    ],
)
def test_rejects_unsafe_interactive_components(
    component: dict[str, object],
) -> None:
    """Reject unbound values, invalid actions and unsafe component properties."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "formSurface",
                "components": [component],
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)


@pytest.mark.parametrize(
    "component",
    [
        {
            "id": "field",
            "component": "TextField",
            "label": "Name",
            "value": {"@path": "/name"},
            "variant": [],
        },
        {
            "id": "picker",
            "component": "ChoicePicker",
            "options": [{"value": "a"}],
            "value": {"@path": "/choice"},
            "variant": [],
        },
        {
            "id": "button",
            "component": "Button",
            "child": "label",
            "action": {"event": {"name": "submit_form"}},
            "variant": [],
        },
    ],
)
def test_rejects_malformed_interactive_enum_values(
    component: dict[str, object],
) -> None:
    """Malformed enum values must raise ValueError, not TypeError."""
    messages = [
        {
            "version": "1.0",
            "updateComponents": {
                "surfaceId": "formSurface",
                "components": [component],
            },
        }
    ]

    with pytest.raises(ValueError):
        validate_a2ui_message_batch(messages)
