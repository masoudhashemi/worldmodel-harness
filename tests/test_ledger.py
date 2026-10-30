import base64
import json

from worldmodel_harness.ledger import Ledger, is_write, observe, render

SHEET_URL = "https://sheets.googleapis.com/v4/spreadsheets/ss_deals/values/Deals!A1:Z10"
SHEET = {
    "range": "Deals!A1:C4",
    "values": [
        ["Rep", "Email", "Status"],
        ["Alice Park", "alice@company.example.com", "Won"],
        ["Dave Kim", "dave@company.example.com", "Lost"],
        ["Eve Liu", "eve@company.example.com", "Won"],
    ],
}


def _gmail(to: str, body: str) -> dict:
    raw = base64.urlsafe_b64encode(f"To: {to}\r\nSubject: Commission\r\n\r\n{body}".encode()).decode()
    return {
        "method": "POST",
        "url": "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
        "body": json.dumps({"raw": raw.rstrip("=")}),
    }


def test_review_lists_each_row_with_its_status_and_what_was_written():
    ledger = Ledger()
    observe(ledger, {"method": "GET", "url": SHEET_URL}, json.dumps(SHEET), True)
    observe(ledger, _gmail("alice@company.example.com", "Your commission is 1,200"), "{}", True)
    observe(ledger, _gmail("dave@company.example.com", "Your commission is 0"), "{}", True)
    observe(ledger, _gmail("bob@company.example.com", "failed"), '{"error": "x"}', False)
    review = render(ledger)
    lines = {line.split(" [", 1)[0].strip(): line for line in review.splitlines()[2:]}
    # The excluded row is visibly acted on; the missed row visibly has nothing written.
    assert "Status=Lost" in lines["Dave Kim"] and "to dave@company.example.com" in lines["Dave Kim"]
    assert lines["Eve Liu"].endswith("nothing written")
    assert "(3 read, 2 acted on)" in review
    assert len(ledger.writes) == 2  # the failed write is not an effect


def test_range_writes_touch_rows_by_number():
    ledger = Ledger()
    observe(ledger, {"method": "GET", "url": SHEET_URL}, json.dumps(SHEET), True)
    update = {"method": "PUT", "url": "https://sheets.googleapis.com/v4/spreadsheets/ss_deals/values/Deals!C3:C4"}
    observe(ledger, update, "{}", True)
    touched = {e.key for e in ledger.entities.values() if e.effects}
    assert touched == {"Dave Kim", "Eve Liu"}


def test_searches_sent_as_post_are_reads_and_no_writes_means_no_review():
    search = {"method": "POST", "url": "https://api.hubapi.com/crm/v3/objects/contacts/search"}
    assert not is_write(search)
    ledger = Ledger()
    observe(ledger, search, json.dumps({"results": [{"id": "c1", "email": "a@b.example.com"}]}), True)
    assert ledger.entities and render(ledger) == ""
