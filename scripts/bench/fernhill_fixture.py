"""Support-bot prompt, tools, and mock tool results shared by the LLM and end-to-end benchmarks."""
from __future__ import annotations

import json

SUPPORT_SYSTEM = """You are Robin, a phone support agent for Fernhill Internet. Fernhill sells email hosting, website hosting, domain registration, and SSL certificates to small businesses.

You are speaking on a phone call. The caller only hears your words, so:
- Keep each reply to one or two short sentences.
- Never use lists, markdown, or symbols. Say numbers as a person would say them.
- Ask for one thing at a time.

Before you discuss an account you must look it up with lookup_customer. Use get_invoices to see billing history. Use issue_refund only after the caller has confirmed the exact invoice and amount.

Policies:
- Duplicate charges are refunded in full.
- Refunds reach the card in five to seven business days.
- Refunds over two hundred dollars need a supervisor, so offer a callback instead of calling issue_refund.
- Never guess account details. If a tool returns no match, ask the caller to repeat the information.
- Do not promise anything these policies do not cover.

Start by answering the caller's question. If you need a tool, call it right away without announcing it."""

CALLER_SYSTEM = """You are role-playing a customer phoning Fernhill Internet support. Stay in character and only speak as the customer.

Fact sheet:
- Name: Dana Reyes
- Account number: AC9077
- Problem: you were charged twice for last month's Email Pro plan, invoice FH-4419 and invoice FH-4420, each $49.99.

Goal: get the duplicate charge refunded and find out when the money will arrive.

Rules:
- Say one or two short sentences per turn, as a real caller would.
- Only give facts when asked. Never invent facts that are not on the sheet.
- If the agent confirms the refund and the timing, thank them and say goodbye."""

TOOLS = [
    {"type": "function", "function": {
        "name": "lookup_customer", "description": "Find a customer by account number.",
        "parameters": {"type": "object", "properties": {
            "account_number": {"type": "string", "description": "Account number such as AC9077"},
            "name": {"type": "string", "description": "Customer full name if given"}},
            "required": ["account_number"]}}},
    {"type": "function", "function": {
        "name": "get_invoices", "description": "List recent invoices for a customer.",
        "parameters": {"type": "object", "properties": {
            "customer_id": {"type": "string", "description": "Customer id from lookup_customer"}},
            "required": ["customer_id"]}}},
    {"type": "function", "function": {
        "name": "issue_refund", "description": "Refund an invoice after the caller confirms it.",
        "parameters": {"type": "object", "properties": {
            "invoice_id": {"type": "string"},
            "amount": {"type": "number", "description": "Refund amount in dollars"},
            "reason": {"type": "string"}},
            "required": ["invoice_id", "amount", "reason"]}}},
]

CUSTOMER = {"customer_id": "cus_1042", "name": "Dana Reyes", "account_number": "AC9077", "plan": "Email Pro", "status": "active"}
INVOICES = [
    {"id": "FH-4419", "amount": 49.99, "status": "paid", "date": "2026-09-30", "description": "Email Pro monthly"},
    {"id": "FH-4420", "amount": 49.99, "status": "paid", "date": "2026-09-30", "description": "Email Pro monthly, duplicate of FH-4419"},
    {"id": "FH-4287", "amount": 49.99, "status": "paid", "date": "2026-08-31", "description": "Email Pro monthly"},
]


def run_tool(name: str, args: dict) -> dict:
    if name == "lookup_customer":
        acct = str(args.get("account_number", "")).upper().replace(" ", "").replace("-", "")
        return dict(CUSTOMER) if acct == "AC9077" else {"error": "no customer found"}
    if name == "get_invoices":
        return {"invoices": INVOICES} if args.get("customer_id") == "cus_1042" else {"error": "unknown customer_id"}
    if name == "issue_refund":
        if args.get("invoice_id") in {i["id"] for i in INVOICES}:
            return {"refund_id": "re_881", "status": "issued", "invoice_id": args["invoice_id"], "amount": args.get("amount")}
        return {"error": "unknown invoice"}
    return {"error": f"unknown tool {name}"}


def tool_message(call_id: str, name: str, args: dict) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(run_tool(name, args))}
