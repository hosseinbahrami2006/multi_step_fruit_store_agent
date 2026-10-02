# Advanced Fruit Store Agent

An educational, multi-step AI agent for managing a fruit-store inventory, preparing invoices, and adapting orders to stock and budget constraints.

The project uses **Ollama** with the local **`qwen2.5:3b`** model. Inventory is stored in a separate JSON file, and all price calculations are performed by a Python calculator tool.

> **Purpose:** Demonstrate planning, tool execution, observation, evaluation, replanning, and decision-making in an AI agent.
>
> **Scope:** A single-user classroom demonstration—not a production checkout system.

---

## Features

- Local language model through Ollama
- JSON-based inventory containing ten fruits
- Product price and stock lookup
- Explicit multi-step order plans
- Deterministic invoice calculations using `Decimal`
- Stock and budget validation
- Quantity reductions when authorized by the user
- Bounded planning attempts and model-call retries
- Short-term conversation memory
- User confirmation before inventory updates
- Execution logs for classroom demonstrations

---

## Agent Architecture

The application separates model-generated proposals from application-controlled execution.

```text
User Request
     |
     v
Interpret Request and Extract Constraints
     |
     v
Route to the Appropriate Workflow
     |
     v
Create and Validate a Plan
     |
     v
Execute a Tool
     |
     v
Observe the Result
     |
     v
Evaluate Against Constraints
     |
     +--> CONTINUE
     +--> REPLAN
     +--> ASK_USER
     +--> FINISH
     +--> AWAIT_CONFIRMATION
```

For order requests, the application enforces this tool sequence:

```text
check_stock → calculate_total → prepare_invoice
```

The model proposes the basket and its quantities. Python validates the plan, executes tools, and checks the results.

### Confirmation Boundary

Inventory updates are handled separately from the model-driven workflow:

```text
Pending Invoice
     |
     v
User types "confirm order"
     |
     v
Recheck Stock and Prices
     |
     v
Update fruits.json
```

The inventory-update function is **not available as an LLM tool**.

---

## Session 3 Concepts

| Concept | Implementation |
|---|---|
| Planner | `FruitAgent.create_plan()` |
| Executor | Registered functions in `TOOLS` |
| Task decomposition | Explicit plan steps |
| Routing | Request interpretation and intent-based workflows |
| ReAct-style feedback | Tool actions, observations, and subsequent decisions |
| Evaluation | `FruitAgent.evaluate()` |
| Replanning | Bounded planning loop |
| Retry handling | Bounded Ollama request retries |
| Continue versus finish | Explicit decision values |
| State | Request, catalog, plan, invoice, history, and status |
| Short-term memory | Recent conversation messages |
| Human approval | Explicit confirmation command |

This is a **constrained planning agent**. It adapts order quantities within a fixed, validated workflow rather than choosing arbitrary tool sequences.

Execution logs show observable actions and results, not the model’s private internal reasoning.

---

## Project Structure

```text
fruit_store_agent/
├── main.py
├── agent.py
├── tools.py
├── calculator.py
├── inventory.py
├── fruits.json
├── requirements.txt
└── README.md
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `main.py` | Command-line interface |
| `agent.py` | Interpretation, routing, planning, evaluation, memory, and execution loop |
| `tools.py` | Inventory lookup, stock checks, invoice tools, and confirmation handler |
| `calculator.py` | Decimal-based invoice calculations |
| `inventory.py` | Inventory validation, reading, and saving |
| `fruits.json` | Product names, prices, and stock |
| `requirements.txt` | Python dependencies |

---

## Requirements

- Python 3.10 or newer recommended
- Ollama installed
- The `qwen2.5:3b` model downloaded
- Sufficient memory to run the model locally

Python dependency:

```text
requests>=2.32.0,<3.0.0
```

Install Ollama from:

https://ollama.com/

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/hosseinbahrami2006/fruit_store_agent.git
cd fruit_store_agent
```

This README describes the advanced implementation. Ensure the files listed above contain that implementation before running it.

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Model

```bash
ollama pull qwen2.5:3b
```

### 5. Start Ollama

If Ollama is not already running:

```bash
ollama serve
```

Keep the service running while using the application.

### 6. Run the Agent

```bash
python main.py
```

---

## Usage

Enter requests in English.

### List Inventory

```text
Which fruits are available?
```

The agent retrieves product names, prices, and available stock.

### Check a Product

```text
What is the price and stock of apples?
```

### Prepare an Invoice

```text
Prepare an invoice for 2 kg of apples and 3 kg of bananas.
```

The agent checks stock, calculates the total, and prepares a quote.

**Preparing a quote does not change inventory.**

### Prepare an Order With a Budget

```text
Prepare an order with 3 kg of apples, 2 kg of bananas,
and 3 kg of oranges. My budget is $25.
Reduce quantities if necessary, but keep all three fruits.
```

When quantity reductions are authorized, the agent can revise the basket after a stock or budget failure.

### Modify a Pending Order

```text
Add one more kilogram of apples.
```

The interpreter receives the pending order and recent conversation as context.

Because interpretation depends on the model, always review the resulting invoice before confirming it.

---

## Commands

Commands are case-insensitive.

| Command | Behavior |
|---|---|
| `confirm order` | Revalidate and purchase the pending order |
| `cancel order` | Discard the pending order without changing stock |
| `reset` | Clear conversation memory, pending order, and execution state |
| `exit` | Close the application |

Use the exact command `confirm order` to approve a purchase. A conversational response such as “yes” does not directly trigger the inventory update.

---

## Example: Planning and Replanning

Assume the inventory includes:

| Fruit | Price per kg | Available stock |
|---|---:|---:|
| Apple | $4.00 | 10 kg |
| Banana | $2.50 | 8 kg |
| Orange | $3.50 | 2 kg |

### User Request

```text
Prepare an order with 3 kg of apples, 2 kg of bananas,
and 3 kg of oranges. My budget is $25.
Reduce quantities if necessary, but keep all three fruits.
```

### Initial Basket

```text
Apples:  3 × $4.00 = $12.00
Bananas: 2 × $2.50 =  $5.00
Oranges: 3 × $3.50 = $10.50

Total: $27.50
```

The initial request exceeds both orange stock and the budget.

### Feedback Loop

```text
PLAN
  Check stock.
  Calculate total.
  Prepare invoice.

ACT
  Run check_stock.

OBSERVE
  Requested oranges: 3 kg.
  Available oranges: 2 kg.

EVALUATE
  Stock constraint failed.

DECIDE
  Replan because reductions are authorized.
```

### One Valid Revised Basket

```text
Apples:  3 × $4.00 = $12.00
Bananas: 2 × $2.50 =  $5.00
Oranges: 2 × $3.50 =  $7.00

Total: $24.00
```

The agent then asks for confirmation.

> Revised quantities may differ between runs. The model proposes them, but application code checks the required products, permitted quantities, stock, and budget.

---

## Inventory Format

`fruits.json` contains a list of products:

```json
[
  {
    "name": "apple",
    "price_per_kg": "4.00",
    "stock_kg": "10"
  },
  {
    "name": "banana",
    "price_per_kg": "2.50",
    "stock_kg": "8"
  }
]
```

### Fields

| Field | Description |
|---|---|
| `name` | Unique product name |
| `price_per_kg` | Price for one kilogram |
| `stock_kg` | Available quantity in kilograms |

Use lowercase, singular product names.

Prices and stock are represented as decimal strings. The application uses Python’s `Decimal` type to avoid binary floating-point errors in financial calculations.

Invoice line totals are rounded to two decimal places using `ROUND_HALF_UP`.

The sample application displays prices using `$`; it does not perform currency conversion.

---

## Tools

### Model-Workflow Tools

| Tool | Purpose |
|---|---|
| `list_inventory` | Return all products |
| `get_product` | Return one product’s information |
| `check_stock` | Validate planned quantities against stock |
| `calculate_total` | Calculate invoice lines and total |
| `prepare_invoice` | Return the calculated invoice for review |

### Application-Only Operation

| Function | Purpose |
|---|---|
| `confirm_pending_order` | Recheck and commit an explicitly confirmed order |

The confirmation function is excluded from the model’s tool registry.

---

## Validation and Decision Rules

### Plan Validation

For an order, the application checks that:

- The required tool sequence is preserved.
- Every requested product appears exactly once.
- No extra products are added.
- No required products are removed.
- Quantities are positive.
- Quantities do not exceed the interpreted request.
- Quantities remain unchanged when reductions are not authorized.

### Result Evaluation

| Result | Decision |
|---|---|
| Stock check passes | Continue |
| Stock check fails and reductions are allowed | Replan |
| Stock check fails without reduction permission | Ask the user |
| Total exceeds budget and reductions are allowed | Replan |
| Total exceeds budget without reduction permission | Ask the user |
| Valid invoice is prepared | Await confirmation |
| Product information is retrieved | Finish |
| Planning attempt limit is reached | Stop with an incomplete result |

If a required fruit has zero stock, retaining it with a positive quantity is impossible. The agent may exhaust its planning attempts and ask for a revised request.

---

## State and Memory

During execution, the agent tracks:

```text
request
catalog
plan
invoice
history
status
```

The agent also keeps:

- Recent conversation messages
- A pending order
- The latest execution state

Memory is stored **only in the running Python process**.

Restarting the application clears conversation memory and pending orders. The JSON inventory remains on disk.

---

## Configuration

The following settings are defined in `agent.py`:

```python
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"

MAX_MODEL_RETRIES = 2
MAX_PLAN_ATTEMPTS = 4
MEMORY_MESSAGES = 12
```

| Setting | Purpose |
|---|---|
| `MODEL` | Ollama model name |
| `OLLAMA_URL` | Local Ollama API endpoint |
| `MAX_MODEL_RETRIES` | Maximum attempts for each model request |
| `MAX_PLAN_ATTEMPTS` | Maximum planning attempts per interpreted request |
| `MEMORY_MESSAGES` | Maximum retained conversation messages |

Despite its name, `MAX_MODEL_RETRIES` is used as the total number of request attempts in the current implementation.

Enable classroom execution logs:

```python
agent = FruitAgent(verbose=True)
```

Disable them:

```python
agent = FruitAgent(verbose=False)
```

---

## Error Handling

The application distinguishes execution problems from constraint failures.

### Model Request Errors

Network errors, invalid JSON responses, and malformed API responses trigger bounded model-request retries.

### Invalid Plans

Plans with invalid tools, products, or quantities are rejected. Validation feedback is supplied to the next planning attempt.

### Business Constraint Failures

Insufficient stock or an over-budget invoice leads to replanning or a request for user clarification.

### Changed Inventory or Prices

Before confirming an order, the application reads inventory again.

If stock is insufficient or the invoice has changed, the quote is rejected and the user must request a new one.

Inventory read and write failures are reported to the user; they do not use the model-request retry mechanism.

---

## Manual Test Scenarios

These are suggested manual checks, not an automated test suite.

| Scenario | Expected Behavior |
|---|---|
| Request all available fruits | Display inventory |
| Request one product’s price | Display product information |
| Order within stock and budget | Prepare quote |
| Order beyond stock without modification permission | Ask for revision |
| Allow reductions when stock is insufficient | Attempt replanning |
| Allow reductions when over budget | Attempt replanning |
| Require a fruit with zero stock | Stop or request revised constraints |
| Confirm a valid quote | Update inventory |
| Cancel a quote | Leave inventory unchanged |
| Change prices before confirmation | Reject stale quote |
| Confirm the same order twice | Second confirmation finds no pending order |
| Stop Ollama before a request | Report failure after bounded attempts |

Back up `fruits.json` before testing purchases, because confirmed orders permanently reduce the stored stock.

---

## Troubleshooting

### Cannot Connect to Ollama

Ensure Ollama is running:

```bash
ollama serve
```

Verify available models:

```bash
ollama list
```

### Model Not Found

Download the configured model:

```bash
ollama pull qwen2.5:3b
```

### Slow Responses

Local inference speed depends on hardware.

Possible options:

- Use hardware acceleration supported by Ollama.
- Close memory-intensive applications.
- Increase the HTTP timeout in `agent.py` if appropriate.

### Invalid Plans or Repeated Planning Failures

Small models can produce invalid structures or misunderstand requests.

Try:

- Explicit quantities
- Clear budget limits
- Explicit permission to reduce quantities
- A simpler request
- A more capable local model

Increasing planning attempts may help, but it does not guarantee success.

### Invalid Inventory File

Check that:

- The file contains valid JSON.
- The top-level value is a list.
- Product names are unique.
- Prices and stock are finite, nonnegative decimal values.
- Each product includes all required fields.

---

## Limitations

- Designed for one running application instance and one user.
- No database transactions or multi-user concurrency control.
- Atomic file replacement prevents partially written JSON, but not concurrent lost updates.
- Conversation memory and pending orders are not persisted.
- Request interpretation depends on the model and may misread user constraints.
- Validation enforces the interpreted request; it cannot guarantee that interpretation matches the user’s original intent.
- There is no optimal basket solver. The planner may fail to find a valid basket even when one exists.
- Initial requested quantities are encouraged through prompting, rather than separately enforced on the first planning attempt.
- No product substitutions, authentication, payment processing, taxes, or delivery calculations.
- No inventory reservation while a quote awaits confirmation.
- No durable order ledger or idempotency key for recovery from uncertain write outcomes.
- No automated test suite is included in this implementation.

For production use, add transactional storage, structured confirmation, durable order records, concurrency controls, and automated tests.

---

## Suggested Future Improvements

- Add typed request and plan schemas.
- Ask the user to approve extracted constraints.
- Use a deterministic quantity-adjustment or optimization tool.
- Add explicit tool-call and overall execution time limits.
- Implement automated tests with mocked model responses.
- Persist conversation state and pending orders.
- Add SQLite or another transactional database.
- Add order IDs and idempotent confirmation.
- Improve handling of impossible constraints.
- Add a web interface.
- Add structured audit logs.

---

## Learning Takeaway

> A plan is a proposal, a tool result is an observation, and evaluation determines the next action.

This project demonstrates how a simple tool-using assistant can become a more capable agent by connecting those elements in a bounded feedback loop.