"""Customer Support AI Agent.

AgentCore entrypoint for the Udacity customer-support project.

Configure the Gateway, Knowledge Base, Memory resource, and AWS region below
before deploying. The Gateway is expected to use the project's temporary NONE
authorizer, so do not expose it to sensitive data or production traffic.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Mapping
from typing import Any

import boto3
from bedrock_agentcore.memory import MemoryClient
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.tools.code_interpreter_client import code_session
from mcp.client.streamable_http import streamable_http_client
from strands import Agent, tool
from strands.hooks import AfterInvocationEvent, HookProvider, HookRegistry, MessageAddedEvent
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from strands_tools.browser import AgentCoreBrowser


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CSAI_Agent")

# Tool-consent prompts are unsuitable for the headless AgentCore runtime.
os.environ.setdefault("BYPASS_TOOL_CONSENT", "true")

# ──  1 — App Initialisation ───────────────────────────────────────────────
app = BedrockAgentCoreApp()

# ──  2 — Configuration ────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-peikor92jh.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID = "DGIWRAZFLA"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-cMLiLhESwe"
MODEL_ID = "us.amazon.nova-2-lite-v1:0"


# ──  3 — Model and Clients ────────────────────────────────────────────────
model = BedrockModel(model_id=MODEL_ID)


# The clients are created lazily so importing this module does not require AWS
# credentials or make network calls.
def _memory_client() -> MemoryClient:
    return MemoryClient(region_name=REGION)


def _bedrock_runtime_client():
    return boto3.client("bedrock-agent-runtime", region_name=REGION)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"<[^<>]+>")
_NAMESPACE_TOKEN_RE = re.compile(r"\{[^{}]*\}")


def _get(value: Any, *names: str, default: Any = None) -> Any:
    """Read the first available key/attribute from SDK dicts or objects."""
    for name in names:
        if isinstance(value, Mapping) and name in value:
            candidate = value[name]
        else:
            candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return default


def _is_configured(value: str) -> bool:
    return bool(value and value.strip() and not _PLACEHOLDER_RE.fullmatch(value.strip()))


def _message_text(message: Any) -> str:
    """Extract text from a Strands message or compatible message object."""
    content = _get(message, "content", default=[])
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, (list, tuple)):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        else:
            block_text = _get(block, "text")
            if isinstance(block_text, str):
                parts.append(block_text)
    return "\n".join(part.strip() for part in parts if part.strip()).strip()


def _message_role(message: Any) -> str:
    role = _get(message, "role")
    if isinstance(role, str):
        return role.lower()
    nested = _get(message, "message", default={})
    role = _get(nested, "role")
    return role.lower() if isinstance(role, str) else ""


def _extract_response_text(response: Any) -> str:
    """Extract final assistant text from a Strands AgentResult or compatible value."""
    if isinstance(response, str):
        return response.strip()

    message = _get(response, "message")
    text = _message_text(message) if message is not None else ""
    if text:
        return text

    text = _message_text(response)
    if text:
        return text
    direct_text = _get(response, "text")
    return direct_text.strip() if isinstance(direct_text, str) else ""


def _remove_exact_duplicate(text: str) -> str:
    """Remove an exact repeated full response or adjacent identical paragraphs."""
    text = text.strip()
    if not text:
        return text

    if len(text) % 2 == 0:
        half = len(text) // 2
        if text[:half].strip() == text[half:].strip():
            return text[:half].strip()

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) > 1:
        unique: list[str] = []
        for paragraph in paragraphs:
            if not unique or paragraph != unique[-1]:
                unique.append(paragraph)
        return "\n\n".join(unique)
    return text


# ---------------------------------------------------------------------------
# AgentCore Memory
# ---------------------------------------------------------------------------


def _strategy_type(strategy: Any) -> str | None:
    """Normalize strategy type from supported SDK response shapes."""
    value = _get(
        strategy,
        "type",
        "strategyType",
        "memoryStrategyType",
        "memory_strategy_type",
    )
    if isinstance(value, Mapping):
        value = _get(value, "type", "value", "name")
    if value is None:
        return None
    return str(value).strip() or None


def _namespace_template(strategy: Any) -> str | None:
    """Return the namespace template declared by an AgentCore Memory strategy."""
    candidates = [
        _get(strategy, "namespaceTemplates"),
        _get(strategy, "namespaces"),
    ]

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

        if isinstance(candidate, (list, tuple)):
            for item in candidate:
                if isinstance(item, str) and item.strip():
                    return item.strip()

    return None


# ──  4 — Namespace Helper ────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> dict[str, str]:
    """Return strategy-type -> namespace-template mappings."""
    response = mem_client.get_memory_strategies(memory_id=memory_id)
    strategies = response

    if isinstance(response, Mapping):
        strategies = _get(
            response,
            "memoryStrategies",
            "strategies",
            "memory_strategies",
            default=[],
        )
    else:
        strategies = _get(response, "memoryStrategies", "strategies", default=response)

    if isinstance(strategies, Mapping):
        strategies = list(strategies.values())
    if not isinstance(strategies, (list, tuple)):
        logger.warning(
            "Unexpected memory-strategy collection: %s",
            type(strategies).__name__,
        )
        return {}

    namespaces: dict[str, str] = {}
    for strategy in strategies:
        kind = _strategy_type(strategy)
        template = _namespace_template(strategy)
        if kind and template:
            namespaces[kind] = template
        elif kind:
            logger.debug("Memory strategy %s has no namespace template", kind)
    return namespaces


def _expand_namespace(
    template: str,
    actor_id: str,
    session_id: str,
    memory_strategy_id: str | None = None,
) -> str:
    """Expand supported AgentCore Memory namespace placeholders."""
    replacements = {
        "{actorId}": actor_id,
        "{actor_id}": actor_id,
        "{sessionId}": session_id,
        "{session_id}": session_id,
    }

    if memory_strategy_id:
        replacements["{memoryStrategyId}"] = memory_strategy_id
        replacements["{memory_strategy_id}"] = memory_strategy_id

    namespace = template
    for token, value in replacements.items():
        namespace = namespace.replace(token, value)

    namespace = namespace.strip()

    if not namespace or "{" in namespace or "}" in namespace:
        return ""

    return namespace


# ──  5 — Memory Hook ─────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Retrieve relevant customer memories and persist completed interactions."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ) -> None:
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        try:
            self.namespaces = get_namespaces(memory_client, memory_id)
        except Exception:
            logger.warning(
                "Unable to load AgentCore Memory strategies; continuing without memory",
                exc_info=True,
            )
            self.namespaces = {}

    def retrieve_customer_context(self, event: MessageAddedEvent) -> None:
        """Prepend retrieved memories to the current user message, if available."""
        messages = getattr(event.agent, "messages", [])
        if not messages:
            return
        message = messages[-1]
        if _message_role(message) != "user":
            return

        query = _message_text(message)
        if not query:
            return

        retrieved: list[str] = []
        for strategy_type, template in self.namespaces.items():
            namespace = _expand_namespace(template, self.actor_id, self.session_id)
            if not namespace:
                logger.warning(
                    "Skipping memory search for strategy %s: namespace template could not be expanded",
                    strategy_type,
                )
                continue

            try:
                response = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )
            except Exception:
                logger.exception("Memory retrieval failed for strategy %s", strategy_type)
                continue

            memories = _get(response, "memories", default=response)
            if not isinstance(memories, (list, tuple)):
                continue

            for memory in memories:
                if isinstance(memory, str):
                    memory_text = memory.strip()
                else:
                    memory_text = _get(memory, "content", "text", "memory", default="")
                    if isinstance(memory_text, Mapping):
                        memory_text = _get(memory_text, "text", default="")
                    memory_text = str(memory_text).strip() if memory_text else ""
                if memory_text:
                    retrieved.append(f"[{strategy_type}] {memory_text}")

        if retrieved:
            original_content = _get(message, "content", default=[])
            replacement = "Customer Context:\n" + "\n".join(retrieved) + "\n\n" + query
            if isinstance(original_content, str):
                message["content"] = replacement
            else:
                message["content"] = [{"text": replacement}]

    def save_support_interaction(self, event: AfterInvocationEvent) -> None:
        """Persist the latest user/assistant pair for this invocation."""
        messages = getattr(event.agent, "messages", [])
        customer_query = ""
        assistant_response = ""

        for message in reversed(messages):
            role = _message_role(message)
            text = _message_text(message)
            if not text:
                continue
            if not assistant_response and role == "assistant":
                assistant_response = text
            elif not customer_query and role == "user":
                customer_query = text
            if customer_query and assistant_response:
                break

        if not customer_query or not assistant_response:
            logger.debug("Skipping Memory write: user or assistant text missing")
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[(customer_query, "USER"), (assistant_response, "ASSISTANT")],
            )
        except Exception:
            logger.exception("Unable to save customer-support interaction")

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ---------------------------------------------------------------------------
# Knowledge Base tool
# ---------------------------------------------------------------------------

# ──  6 — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search product and support content in the configured Bedrock Knowledge Base.

    Args:
        query: Product, policy, warranty, loyalty, or order-status question.

    Returns:
        Relevant retrieved text, or a clear message if retrieval is unavailable.
    """
    if not _is_configured(KB_ID):
        return "Knowledge base is not configured."

    try:
        response = _bedrock_runtime_client().retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception:
        logger.exception("Knowledge Base retrieval failed")
        return "I couldn't retrieve information from the knowledge base right now."

    chunks: list[str] = []
    for result in response.get("retrievalResults", []):
        content = result.get("content", {})
        text = content.get("text") if isinstance(content, Mapping) else None
        if isinstance(text, str) and text.strip():
            chunks.append(text.strip())
    return "\n---\n".join(chunks) if chunks else "No relevant information was found in the knowledge base."


# ---------------------------------------------------------------------------
# Loyalty discount tool
# ---------------------------------------------------------------------------

# ──  7 — Loyalty Discount Tool ───────────────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """Calculate points redemption, tier discount, final total, and points earned.

    Points redeem in 500-point increments for $5, capped at 50% of subtotal.
    Tier discount is applied after point redemption.

    Args:
        loyalty_points: Current customer points balance.
        tier: Silver, Gold, or Platinum.
        order_total: Order subtotal in USD.
        product_category: standard, device, or fresh.

    Returns:
        JSON calculation breakdown.
    """
    points = max(0, int(loyalty_points))
    subtotal = max(0.0, float(order_total))
    normalized_tier = str(tier).strip().title()
    category = str(product_category).strip().lower()

    code = f"""
import json
import math
loyalty_points = {points}
tier = {json.dumps(normalized_tier)}
order_total = {subtotal}
product_category = {json.dumps(category)}
earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}
tier_rate = tier_rates.get(tier, 0.0)
earn_rate = earn_rates.get(product_category, earn_rates["standard"])
redeemable_points = max(0, loyalty_points)
points_redeemed = min(
    (redeemable_points // 500) * 500,
    int((order_total * 0.50) // 5) * 500,
)
points_discount = points_redeemed / 100.0
subtotal_after_points = max(0.0, order_total - points_discount)
tier_discount = subtotal_after_points * tier_rate
final_total = max(0.0, subtotal_after_points - tier_discount)
total_savings = order_total - final_total
points_earned = int(math.floor(final_total * earn_rate))
remaining_points = loyalty_points - points_redeemed + points_earned
result = {{
    "order_total": round(order_total, 2),
    "points_redeemed": points_redeemed,
    "points_discount": round(points_discount, 2),
    "tier": tier,
    "tier_discount_rate": tier_rate,
    "tier_discount": round(tier_discount, 2),
    "final_total": round(final_total, 2),
    "total_savings": round(total_savings, 2),
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as session:
            response = session.invoke(
                "executeCode",
                {"code": code, "language": "python", "clearContext": True},
            )
        events = _get(response, "content", default=[])
        if isinstance(events, (list, tuple)):
            for event in events:
                if _get(event, "type") == "text":
                    result_text = _get(event, "text")
                    if isinstance(result_text, str) and result_text.strip():
                        return result_text.strip()
        raise RuntimeError("Code Interpreter returned no text result")
    except Exception:
        logger.exception("Code Interpreter unavailable; using tier-only fallback")
        rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        rate = rates.get(normalized_tier, 0.0)
        discount = subtotal * rate
        return json.dumps(
            {
                "calculation": "tier_discount_fallback",
                "tier": normalized_tier,
                "tier_discount_rate": rate,
                "tier_discount": round(discount, 2),
                "final_total": round(max(0.0, subtotal - discount), 2),
                "note": "Points redemption and points accrual were not calculated.",
            }
        )


# ---------------------------------------------------------------------------
# AgentCore entrypoint
# ---------------------------------------------------------------------------



SYSTEM_PROMPT = """\
You are an intelligent e-commerce customer support assistant.

Help customers with order tracking, refund requests, product and policy questions,
and loyalty calculations.

Tool usage requirements:
- For questions about loyalty tiers, tier benefits, product information, or support
  policies, call search_knowledge_base before answering.
- Base catalog, loyalty-tier, and policy answers on the retrieved Knowledge Base
  content. Do not claim the Knowledge Base is unavailable unless the retrieval tool
  actually reports a failure.
- Answer the specific question asked. Include only information relevant to that
  question; do not append unrelated loyalty-point rules, redemption details, or
  other topics unless the customer asks for them.
- For questions about loyalty-tier benefits, summarize the benefits explicitly
  returned by the Knowledge Base and do not substitute points-earning rules for
  tier benefits.
- For order tracking and refund requests, use the appropriate available tools to
  obtain authoritative information.
- For loyalty calculations, use the calculation tool when appropriate.
- Do not invent order details, refund approvals, tracking numbers, policy terms,
  or loyalty results.
- If retrieval returns no relevant information, say that the available sources do
  not contain enough information to answer. Do not fabricate an answer.

Keep customer information private. Use retrieved customer context only when it is
relevant to the current request. Explain tool results clearly and concisely.
""".strip()




def _configured_resources() -> bool:
    missing = [
        name
        for name, value in (
            ("GATEWAY_URL", GATEWAY_URL),
            ("KB_ID", KB_ID),
            ("REGION", REGION),
            ("MEMORY_ID", MEMORY_ID),
        )
        if not _is_configured(value)
    ]
    if missing:
        logger.error("Configuration placeholders remain: %s", ", ".join(missing))
        return False
    return True


def _aws_credentials_available() -> bool:
    """Return False when the environment lacks AWS credentials for Bedrock calls."""
    try:
        session = boto3.Session()
        return session.get_credentials() is not None
    except Exception:
        return False


# ──  8 — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload: dict[str, Any], context: Any = None) -> str:
    """Handle one customer-support request in AgentCore Runtime."""
    try:
        user_input = payload.get("prompt")
        if not isinstance(user_input, str) or not user_input.strip():
            return "Please provide a customer-support question in the 'prompt' field."

        if not _configured_resources():
            return "Customer support resources are not configured. Please try again later."
        if not _aws_credentials_available():
            logger.warning("AWS credentials are not configured in this environment.")
            return "AWS credentials are not configured. Please configure AWS credentials or run this in an AWS-enabled environment."

        actor_id = str(payload.get("customer_id") or "anonymous").strip()
        session_id = str(payload.get("session_id") or uuid.uuid4()).strip()
        if not actor_id:
            actor_id = "anonymous"
        if not session_id:
            session_id = str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=_memory_client(),
            memory_id=MEMORY_ID,
        )
        browser = AgentCoreBrowser(region=REGION)
        tools: list[Any] = [search_knowledge_base, calculate_loyalty_discount, browser.browser]

        mcp_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()
            tools.extend(gateway_tools)
            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=SYSTEM_PROMPT,
            )
            response = await agent.invoke_async(user_input.strip())
            return _remove_exact_duplicate(_extract_response_text(response))

    except Exception as exc:
        message = str(exc)
        if "Unable to locate credentials" in message or "NoCredentialsError" in message:
            logger.warning("AWS credentials are missing while invoking the customer support agent.")
            return "AWS credentials are not configured. Please configure AWS credentials or run this in an AWS-enabled environment."
        logger.exception("Customer-support invocation failed")
        return "I encountered an unexpected error while handling your request. Please try again."


# ---------------------------------------------------------------------------
# Local and AgentCore entrypoints
# ---------------------------------------------------------------------------


def main() -> None:
    """Run AgentCore Runtime or one local JSON request."""
    parser = argparse.ArgumentParser(description="Run the customer-support agent.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run one local JSON request instead of starting AgentCore Runtime.",
    )
    parser.add_argument(
        "payload",
        nargs="?",
        help="JSON request payload; if omitted, read JSON from stdin.",
    )
    args = parser.parse_args()

    if not args.local:
        app.run()
        return

    raw_payload = args.payload if args.payload is not None else sys.stdin.read()
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid JSON payload: {exc}")
    if not isinstance(payload, dict):
        parser.error("JSON payload must be an object containing a 'prompt' field.")

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = asyncio.run(invoke(payload))
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted.\n")
    print(_remove_exact_duplicate(_extract_response_text(result) if not isinstance(result, str) else result))


if __name__ == "__main__":
    # app.run()
    # Uncomment the line below and comment app.run() for local CLI testing:
    main()
