# Project: Building a Customer Support AI Agent with Amazon Bedrock AgentCore

**Udacity — AWS AI Engineering Nanodegree**

---

## Overview

In this project you will build a functional AI customer support agent for a fictional Amazon store. Starting from a simple local chatbot, you will progressively add cloud infrastructure, external tool integration, a knowledge base, persistent memory, a code interpreter, and a browser — finishing with a deployable agent that can handle customer inquiries end-to-end.

By the end of the project your agent will be able to:

- Answer questions about products, return policies, and loyalty rewards using Retrieval-Augmented Generation (RAG)
- Look up order status and process refunds by calling Lambda functions through the AgentCore Gateway
- Remember customer preferences and conversation history across multiple sessions
- Calculate exact loyalty discounts using a secure code sandbox
- Navigate websites to fetch live information

---

## Learning Objectives

After completing this project you will be able to:

1. Deploy an AI agent to Amazon Bedrock AgentCore
2. Wire up API Gateway and Lambda tools via the AgentCore Gateway using the Model Context Protocol (MCP)
3. Implement RAG with a Bedrock Knowledge Base
4. Add short-term (session) and long-term (cross-session) memory using AgentCore Memory
5. Use the AgentCore Code Interpreter for precise computation
6. Integrate the AgentCore Browser tool for live web access

---

## Prerequisites

### AWS Account

- An active AWS account with permission to create and manage:
  - IAM roles and policies
  - Lambda functions
  - API Gateway REST APIs
  - Amazon Bedrock Managed Knowledge Bases (with S3 access)
  - Amazon Bedrock AgentCore resources (Runtime, Gateway, Memory)
- All resources should be created in **us-east-1** (N. Virginia) unless stated otherwise.

### Local Development Environment

| Tool | Version |
|------|---------|
| Python | 3.13+ |
| [uv](https://docs.astral.sh/uv/) | Latest |
| AWS CLI | v2 |
| AgentCore CLI (`agentcore`) | Installed via `bedrock-agentcore-starter-toolkit` |
| Node.js (for MCP Inspector) | 18+ |

### Model Access

Enable the following model in the Amazon Bedrock console under **Model access**:

- **Amazon Nova 2 Lite** (`amazon.nova-2-lite-v1:0`). The starter code invokes its global inference profile, `global.amazon.nova-2-lite-v1:0`.

> **CLI compatibility:** This released project intentionally uses the Python-based
> Bedrock AgentCore Starter Toolkit CLI. AWS recommends the newer npm-based
> AgentCore CLI for new projects, but its project format and commands differ from
> this project. Do not install both CLIs in the same environment because both
> provide an `agentcore` command.

---

## Project Structure

```
project/
└── starter/
    ├── main.py                  ← your starting point (fill in the TODOs)
    ├── setup_permissions.py     ← run after deployment to configure the agent role
    ├── pyproject.toml           ← Python dependencies
    ├── product_catalog.txt      ← upload to the Knowledge Base
    └── lambda/
        ├── order_tracker.py     ← deploy as-is
        ├── refund_processor.py  ← deploy as-is
        └── lambda_schema        ← refund tool schema
```

---

## Part 1 — AWS Infrastructure Setup

Complete these steps **before** writing any agent code.

### Step 1.1 — Project Initialisation

```bash
# From the repository root
cd starter
uv sync --python 3.13
```

Run the `agentcore` commands below from `starter/` using `uv run agentcore`
if your virtual environment is not activated.

### Step 1.2 — Deploy the Lambda Functions

The two Lambda functions (`order_tracker.py` and `refund_processor.py`) are provided in `starter/lambda/`. Deploy them to AWS Lambda before proceeding.

1. In the AWS Lambda console, create two new functions (Python 3.12 runtime):
   - `order-tracker`
   - `refund-processor`
2. Paste the contents of each file into the inline code editor (or zip and upload).
3. Attach an execution role with basic Lambda permissions (CloudWatch Logs).
4. Note the ARN of each function — you will need them in the next step.

### Step 1.3 — Set Up API Gateway and AgentCore Gateway

The two provided Lambda functions use different integrations:

- `order-tracker` expects an API Gateway proxy event containing `resource`,
  `httpMethod`, and `pathParameters`.
- `refund-processor` expects direct AgentCore Gateway tool arguments and reads
  the selected tool name from the Lambda client context.

Configure them as follows.

#### A. Expose `order-tracker` through API Gateway

1. In **API Gateway**, create a REST API.
2. Create these resources and methods:
   - `GET /orders/{order_id}`
   - `GET /customers/{customer_id}/orders`
   - `GET /customers/{customer_id}`
3. Configure every method as a **Lambda proxy integration** with the
   `order-tracker` Lambda function.
4. Give the operations unique operation names, such as `get_order`,
   `get_customer_orders`, and `get_customer`. These become MCP tool names.
5. Deploy the REST API to a stage, such as `prod`.

#### B. Create the AgentCore Gateway and targets

1. Open **Amazon Bedrock** → **AgentCore** → **Gateways**.
2. Create `CustomerSupportGateway` with the **NONE** authorizer. The starter's
   MCP connection is unsigned, so another authorizer will reject it.
3. Add the deployed REST API stage as an **API Gateway target** named
   `order_tracker`, exposing the three GET methods above.
4. Add `refund-processor` as a **Lambda target** named `refund_processor` and
   import `starter/lambda/lambda_schema` as its tool schema.
5. Copy the Gateway URL ending in `/mcp` into `GATEWAY_URL` in `main.py`.

> The NONE authorizer is used only to keep this sandbox project focused on tool
> integration. Do not use it for a production Gateway, do not send sensitive
> data through it, and delete the Gateway after completing the project.

**Verify with MCP Inspector:**
```bash
npx @modelcontextprotocol/inspector
# Connect to your Gateway URL and confirm the three order tools and three
# refund tools are listed. Names may be prefixed as target_name___tool_name.
```

### Step 1.4 — Create the Knowledge Base

1. Upload `starter/product_catalog.txt` to an **S3 bucket** in your account.
2. In **Amazon Bedrock AgentCore → Built-in tools → Knowledge Base**, choose
   **Create Managed Knowledge Base**:
   - Name: `CustomerSupportKB`
   - Embedding model type: **Managed**
   - Service role: let the console create a new role
   - Data source: the S3 bucket and `product_catalog.txt` from above
   - Use the default encryption settings
3. **Sync** the data source and wait for the sync to complete.
4. Copy the **Knowledge Base ID** — paste it into `KB_ID` in your `main.py`.

**Verify:**
```bash
# In the console, use the Knowledge Base "Test" tab
# Query: "What is the return policy for electronics?"
# Expected: 15-day return window for electronics
```

### Step 1.5 — Create the AgentCore Memory Resource

1. In the Bedrock console → **AgentCore** → **Memory**, create a new Memory resource:
   - Name: `CustomerSupportMemory`
2. Add two **Memory Strategies**:

   | Strategy | Name | Namespace |
   |---|---|---|
   | Semantic extraction | `customer_facts` | `cs_agent/{actorId}/facts` |
   | User preference | `customer_preferences` | `cs_agent/{actorId}/preferences` |

3. Wait until the memory is **ACTIVE**, then copy its **Memory ID** into `MEMORY_ID` in `main.py`. If an initial invocation reports that memory is not active, wait a few minutes and retry.

---

## Part 2 — Building the Agent

Open `starter/main.py`. It contains scaffolding and `# TODO` comments marking every section you need to implement. Work through the TODOs in order.

### Section 1 — Configuration and Initialisation

Fill in your resource IDs and set up:
- `BedrockAgentCoreApp`
- `BedrockModel` with Amazon Nova 2 Lite
- `MemoryClient` and `boto3` Bedrock runtime client

### Section 2 — Knowledge Base Tool

Implement `search_knowledge_base(query)`:
- Call the Bedrock Knowledge Base Retrieve API
- Join result chunks with `"\n---\n"`

**Test:**
```bash
agentcore invoke '{"prompt": "Is the Kindle Paperwhite waterproof?"}'
# Expected: mention of IPX8 rating
```

### Section 3 — Long-Term Memory Hook

Implement `MemoryHook` with two methods:
- `retrieve_customer_context` — query all memory namespaces and prepend results to the user message
- `save_support_interaction` — save the completed (user, assistant) turn after each response

When reading a strategy's namespace, use `namespaceTemplates[0]` and fall back
to the legacy `namespaces[0]` field when needed.

### Section 4 — Loyalty Discount Tool (Code Interpreter)

Implement `calculate_loyalty_discount(loyalty_points, tier, order_total, product_category)`:
- Build a Python code string containing the discount logic
- Execute it with `code_session()` and return the JSON result
- Include a fallback for when the Code Interpreter is unavailable

**Test:**
```bash
agentcore invoke '{"prompt": "I am a Gold member with 4250 points. Calculate my discount on a $150 order.", "customer_id": "CUST-123", "session_id": "s1"}'
```

### Section 5 — Main Entrypoint

Implement the `invoke(payload, context)` function:
- Extract `prompt`, `customer_id`, and `session_id` from the payload
- Instantiate `MemoryHook` and `AgentCoreBrowser`
- Connect to the Gateway via `MCPClient` and load gateway tools
- Build the `Agent` with all tools and hooks and return its response

### Section 6 — Deploy to AgentCore

```bash
# Configure the Starter Toolkit CLI (first time only)
agentcore configure --entrypoint main.py --name <your-agent-name> --deployment-type direct_code_deploy --runtime PYTHON_3_13 --disable-memory

# Deploy the agent
agentcore deploy
```

`--disable-memory` disables only the toolkit's automatic memory creation; your
agent uses the memory you created in Step 1.5. Let the toolkit create the runtime
execution role when prompted.

After deployment, run this from `starter/` using your student AWS credentials.
Make sure `KB_ID`, `MEMORY_ID` and `REGION` are filled in as strings in `main.py`:

```bash
uv run setup_permissions.py
```

This grants the agent access to your KB, memory and browser. Rerun it if you
change your resource IDs or execution role.

Wait briefly for the policy to take effect, then invoke the deployed agent:

```bash
agentcore invoke '{"prompt": "Hello, what can you help me with?", "customer_id": "CUST-123", "session_id": "test-1"}'
```

---

## Part 3 — Functional Testing

Run the following test scenarios and verify the expected behaviour. Include screenshots or copy the terminal output in your submission.

### Test 1 — Order Tracking

```bash
agentcore invoke '{"prompt": "Can you track order ORD-001?", "customer_id": "CUST-123", "session_id": "t1"}'
# Expected: shipping status, tracking number TRK987654321, carrier UPS, estimated delivery
```

### Test 2 — Refund Processing

```bash
agentcore invoke '{"prompt": "I want to return my Kindle Paperwhite (ORD-002). Please initiate a refund.", "customer_id": "CUST-123", "session_id": "t2"}'
# Expected: refund ID, APPROVED status, 3-5 business days message
```

### Test 3 — Knowledge Base (RAG)

```bash
agentcore invoke '{"prompt": "What are the benefits of the Platinum loyalty tier?", "customer_id": "CUST-123", "session_id": "t3"}'
# Expected: free same-day shipping, 15% discount, priority support
```

### Test 4 — Memory (Long-Term)

```bash
# Session A — introduce yourself
agentcore invoke '{"prompt": "Hi, I am Jane. I prefer concise responses.", "customer_id": "CUST-123", "session_id": "s-A"}'

# Wait at least 30 seconds for memory extraction.

# Session B (new session) — verify recall
agentcore invoke '{"prompt": "Do you remember my name and communication preference?", "customer_id": "CUST-123", "session_id": "s-B"}'
# Expected: agent recalls "Jane" and "concise responses"
```

### Test 5 — Loyalty Discount Calculation

```bash
agentcore invoke '{"prompt": "I am a Gold member with 4250 points. Calculate my discount on a $150 standard order.", "customer_id": "CUST-123", "session_id": "t5"}'
# Expected: points redeemed, tier discount 10%, final total, remaining points
```

### Test 6 — Browser Tool

```bash
agentcore invoke '{"prompt": "Go to https://www.udacity.com and tell me the page title.", "customer_id": "CUST-123", "session_id": "t6"}'
# Expected: page title retrieved from live Udacity.com
```

---

## Submission Checklist

- [ ] `main.py` with all TODOs completed
- [ ] Screenshots or terminal output for all 6 test scenarios
- [ ] Brief written reflection (200–400 words) covering:
  - One design decision you made and why
  - One challenge you encountered and how you solved it
  - How you would extend this agent for a production environment

---

## Helpful References

- [Amazon Bedrock AgentCore Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
- [Strands Agents Documentation](https://strandsagents.com)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector)
- [uv Package Manager](https://docs.astral.sh/uv/)
