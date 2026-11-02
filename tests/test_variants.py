import json

from worldmodel_harness.variants import apply, leaks, make_variant

ROW = {
    "example_id": 1,
    "answer": "",
    "prompt": [{"role": "system", "content": "sys ss_badges"}, {"role": "user", "content": "Check ss_badges for Bob Chen."}],
    "info": json.dumps({
        "task_name": "operations.badges",
        "zapier_tools": ["google_sheets_get_many_rows"],
        "initial_state": {"google_sheets": {"spreadsheets": [{"id": "ss_badges", "rows": [
            {"Contractor": "Bob Chen", "Badge": "BDG-3001", "Email": "bob@acme.example.com", "Phone": "+15553001001"},
            {"Contractor": "Eve Liu", "Badge": "BDG-3002", "Email": "eve@acme.example.com", "Phone": "+15553002002"},
        ]}]}},
        "assertions": [
            {"type": "twilio_sms_sent", "to_number": "+15553001001", "body_contains": "BDG-3001"},
            {"type": "gmail_message_sent", "to": "payroll@acme.example.com", "body_contains": ["Bob", "Eve"]},
        ],
    }),
}


def test_variant_renames_every_surface_literal_consistently():
    variant, mapping = make_variant(ROW, 1)
    info = json.loads(variant["info"])
    assert not leaks(variant, mapping)
    assert info["task_name"] == "operations.badges~v1" and info["zapier_tools"] == ["google_sheets_get_many_rows"]
    row = info["initial_state"]["google_sheets"]["spreadsheets"][0]["rows"][0]
    assert info["assertions"][0] == {
        "type": "twilio_sms_sent", "to_number": row["Phone"], "body_contains": row["Badge"]
    }
    assert info["assertions"][1]["body_contains"] == [mapping["Bob"], mapping["Eve"]]
    assert mapping["Bob"] != mapping["Eve"] and row["Contractor"].startswith(mapping["Bob"])
    assert variant["prompt"][0]["content"] == "sys ss_badges"  # the system prompt is untouched
    assert mapping["ss_badges"] in variant["prompt"][1]["content"]
    assert make_variant(ROW, 1)[1] == mapping and make_variant(ROW, 2)[1] != mapping


def test_apply_replaces_whole_literals_only():
    mapping = {"ss_badges": "ss_x", "BDG-3001": "BDG-9"}
    assert apply("ss_badges_archive BDG-30012 <@ss_badges> BDG-3001.", mapping) == "ss_badges_archive BDG-30012 <@ss_x> BDG-9."
