# The Price Is Right — Amazon Product Price Predictor

Predicts how much a product costs, in dollars, purely from its text description — scraped
from ~2.8M Amazon listings. This was a three-week capstone project that escalates through
five different modeling strategies, ending in an autonomous multi-agent app that scans the
web for underpriced deals in real time and pushes phone notifications when it finds one.

The project is organized as three phases, each in its own folder:

| Folder | Phase | What happens |
|---|---|---|
| `week6/` | Data + traditional ML + frontier fine-tuning | Curate the dataset, build an evaluation harness, try everything from random guessing to XGBoost to a from-scratch neural net to fine-tuning GPT-4.1-nano |
| `week7/` | Open-source fine-tuning | Fine-tune Meta's Llama 3.2 3B with QLoRA (4-bit + LoRA adapters) — beats every frontier model tried zero-shot |
| `week8/` | Agentic deployment | Deploy the fine-tuned Llama as a serverless GPU endpoint (Modal), wrap it in a multi-agent system with RAG, deal-scanning, and push notifications, serve it all through a Gradio app |

The throughline metric across all three weeks is **average absolute dollar error** on a
held-out test set of real Amazon listings.

---

## 1. The core idea

Every model in this project is graded the same way, using `Tester`/`evaluate()` in
`pricer/evaluator.py` (week6/week7) — a harness that:
- runs a `predictor(item) -> float` function over `size` test items (default 200),
- prints each prediction live, color-coded **green** (error < $40 or <20% of price),
  **orange** (error < $80 or <40%), or **red** (worse),
- reports average error, MSE, R², a scatter plot of predicted vs. actual price, and a
  running-mean-error chart with a confidence band.

Because every model — from a coin-flip baseline to a fine-tuned LLM — is scored through this
same harness, the results are directly comparable, and the project becomes a clean ladder of
"how much better can we do."

---

## 2. Week 6 — Data curation, baselines, and frontier fine-tuning

### Day 1 — Curate the dataset
- Source: `McAuley-Lab/Amazon-Reviews-2023` on HuggingFace, 8 categories (Electronics, Automotive,
  Appliances, Toys and Games, Office Products, Tools and Home Improvement, Cell Phones and
  Accessories, Musical Instruments), loaded in parallel with a multiprocess `ItemLoader`
  (`pricer/loaders.py`).
- Each raw listing is turned into an `Item` object (`pricer/items.py`) by `pricer/parser.py`:
  - keep only items priced **$0.50–$999.49**,
  - concatenate title + description + features + details into one text blob, strip noisy
    fields (part numbers, batteries-included flags, etc.) and long alphanumeric product codes,
  - drop anything under 600 characters of clean text, cap the rest at 4000 characters,
  - parse an "Item Weight" field into pounds where present.
- Dedupe by title and by full text, then resample to **820,000 items** using price-weighted
  sampling (weight ∝ price²) while downweighting over-represented categories
  (Tools & Home Improvement ×0.5, Automotive ×0.05).
- Split: **800,000 train / 10,000 val / ~10,000 test** (plus a "lite" 20K/1K/1K version for
  fast iteration), pushed to the HF Hub.

### Day 2 — Clean up the text with an LLM
Raw scraped listings are noisy, so every item's text is rewritten into a standardized
`Title / Category / Brand / Description / Details` format by an LLM
(`groq/openai/gpt-oss-20b`, low reasoning effort). Doing this for 800K items one at a time
would be slow and expensive, so `pricer/batch.py` batches items in groups of 1,000, submits
them through **Groq's batch API** (24-hour completion window), and polls for results — the
full 800K-item run cost about **$30**. Output becomes each item's `summary` field.

### Day 3 — Baselines and traditional ML
All of these are scored with the same `evaluate()` harness:
1. **Random** — a random number 1–1000 (pure noise floor).
2. **Constant** — always guesses the training-set mean price.
3. **Linear regression** — on hand-engineered features: weight, "weight unknown" flag, text length.
4. **Bag-of-words + linear regression** — `CountVectorizer(max_features=2000)` on the summary text.
5. **Random Forest** — 100 trees on the bag-of-words features (trained on the full 800K items in
   ~15 hours, reaching $56.40 error; a 15K-item quick version is used in the notebook itself).
6. **XGBoost** — 1000 trees, same features, faster and slightly better than Random Forest.

### Day 4 — Deep learning and frontier LLMs, zero-shot
- **Human baseline** — the author manually guessed prices for 100 items, logged for comparison.
- **From-scratch PyTorch network** — an 8-layer feedforward net (5000→128→64×6→1) on a
  5000-feature hashing vectorizer, trained 2 epochs with Adam and MSE loss.
- **Frontier LLMs, zero-shot, no training at all** — just prompted with "Estimate the price of
  this product...": GPT-4.1-nano, Claude Opus 4.5, Gemini 3 Pro, Gemini 3.1 Flash Lite,
  Grok 4.1 Fast, GPT-5.1. Several of these zero-shot frontier models beat every traditional-ML
  baseline, which is the "aha" of this day.

### Day 5 — Fine-tune a frontier model (OpenAI)
- Training data is turned into JSONL chat-format pairs:
  ```json
  {"messages": [
    {"role": "user", "content": "Estimate the price of this product. Respond with the price, no explanation\n\n<summary>"},
    {"role": "assistant", "content": "$<price>"}
  ]}
  ```
- Fine-tuned `gpt-4.1-nano-2025-04-14` via the OpenAI Fine-Tuning API
  (`n_epochs=1, batch_size=1, seed=42`). The notebook experiments with training-set size —
  results improve from ~$97 error at 200 examples down to **~$68–76** at 20,000 examples,
  showing the fine-tune scales with data even though OpenAI's own guidance suggests 50–100
  examples is "enough."

### `redemption_train.ipynb` / `redemption_run.ipynb` — a bigger neural net
A much larger, residual-block deep neural network (`pricer/deep_neural_network.py`,
**~289 million parameters**): 8 residual blocks of width 4096, LayerNorm + Dropout(0.2), L1
loss (trains directly on MAE), AdamW + cosine annealing LR schedule, gradient clipping.
Targets are log-transformed and z-score normalized before training and inverted at inference
time. Trained 5 epochs (~4 hours on an M1 Mac), finishing at **$53.15 val MAE** — the best
model in week6. Weights are saved to `deep_neural_network.pth` so `redemption_run.ipynb` can
just load and evaluate without retraining.

**Week 6 leaderboard (dollars, lower is better):**

| Model | Error |
|---|---|
| Random | $106.18 |
| Linear regression | $101.56 |
| Human | $87.62 |
| Bag-of-words + linear regression | $76.81 |
| Fine-tuned GPT-4.1-nano | $75.91 |
| Random Forest | $72.28 |
| XGBoost | $68.23 |
| Zero-shot GPT-4.1-nano | $62.51 |
| Grok 4.1 Fast | $57.62 |
| Gemini 3 Pro | $50.54 |
| Deep Neural Network (residual) | $46.49 |
| Claude 4.5 Sonnet | $47.10 |
| GPT-5.1 | $44.74 |

---

## 3. Week 7 — Fine-tune an open-source model (Llama 3.2 3B + QLoRA)

The actual GPU-heavy fine-tuning runs in **Google Colab** (linked from the day1/day3&4/day5
notebooks). Those three Colab notebooks have been pulled down and saved into
`week7/colab_notebooks/` in this project, so the full training code is now local too:

- **Day 2 — Prompt data prep** (the substantial local notebook): tokenize every item's
  `summary` with the `meta-llama/Llama-3.2-3B` tokenizer, find that a **110-token cutoff**
  covers the vast majority of items, truncate anything longer, and build the exact
  training string format:
  ```
  What does this cost to the nearest dollar?

  <summary, truncated to 110 tokens>

  Price is $<rounded price>.00
  ```
  Test-set items keep the *unrounded* price as the target instead, so evaluation error is
  measured against the true price. The prompt/completion pairs are pushed to the Hub as
  `items_prompts_full` / `items_prompts_lite`.

- **`colab_notebooks/day1_lora_qlora_intro.ipynb`** — walks through loading the base model
  with no quantization, 8-bit, and 4-bit (NF4, double-quant, `bnb_4bit_compute_dtype=bfloat16`)
  to compare memory footprint, then hand-computes LoRA adapter parameter counts for both the
  lite config (r=32, attention layers only, ~1.2M params) and the full config (r=256,
  attention + MLP layers, ~151.7M params / ~607MB).

- **`colab_notebooks/day3_4_training.ipynb`** — the actual fine-tune. Exact hyperparameters:

  | Param | Lite mode | Full mode |
  |---|---|---|
  | Epochs | 1 | 3 |
  | Batch size (per device) | 32 | 256 |
  | Max sequence length | 128 tokens | 128 tokens |
  | Gradient accumulation steps | 1 | 1 |
  | LoRA r | 32 | 256 |
  | LoRA alpha | 64 (`r*2`) | 512 (`r*2`) |
  | LoRA dropout | 0.1 | 0.1 |
  | Target modules | `q_proj,v_proj,k_proj,o_proj` | + `gate_proj,up_proj,down_proj` |
  | Learning rate | 1e-4 | 1e-4 |
  | LR scheduler | cosine, warmup ratio 0.01 | cosine, warmup ratio 0.01 |
  | Weight decay | 0.001 | 0.001 |
  | Optimizer | `paged_adamw_32bit` | `paged_adamw_32bit` |
  | Quantization | 4-bit NF4, double-quant, bf16 compute (A100) / fp16 (T4) | same |
  | Val size | 500 | 1000 |

  Trained with `trl`'s `SFTTrainer` + `SFTConfig`, `max_grad_norm=0.3`, `group_by_length=True`,
  pushed to the HF Hub every `save_steps` via `hub_strategy="every_save"`, logged to
  Weights & Biases. **No `set_seed()` call happens before `fine_tuning.train()`** — see the
  reproducibility note below.

- **`colab_notebooks/day5_eval.ipynb`** — loads the base model + the pushed LoRA adapter
  (pinned to a specific HF Hub commit `revision`) and runs the same `evaluate()` harness
  from `week7/util.py` against the test set, with `set_seed(42)` called only at
  **inference** time (for deterministic generation), not during training.

### Reproducibility — will retraining give you the same weights?

**No — not bit-identical weights, even with everything else matched.** Looking at the actual
training notebook: `set_seed` is imported but **never called before `fine_tuning.train()`**,
so the LoRA adapter's initial random weights, the data-loader's shuffle order, and dropout
sampling are all unseeded. On top of that:
- GPU non-determinism: cuDNN/attention kernels used by `bitsandbytes`/`transformers` are not
  bit-deterministic by default, even if you did seed everything.
- Colab hands you whatever GPU is free (T4 vs A100), which changes `use_bf16` and numeric
  precision, which changes the trained weights.
- Package versions drift (`bitsandbytes==0.48.2 trl==0.25.1` are pinned in the install cell,
  but `transformers`/`peft`/`torch` are whatever Colab has installed that day).

So a retrain will get you **a model with similar performance** (same architecture, same data,
same hyperparameters) but **different weight tensors** than the original run.

**The good news: you don't need to retrain to get "the same" weights.** The actual fine-tuned
LoRA adapter from the original run is already sitting on the HuggingFace Hub, pinned to an
exact commit:
```
meta-llama/Llama-3.2-3B  (base model)
+ PeftModel adapter: ed-donner/price-2025-11-28_18.47.07
  revision: b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd   (full-dataset run, $39.85 error)
+ lite run: ed-donner/price-2025-11-30_15.10.55-lite
```
**Caveat:** those are the course author's (`ed-donner`'s) own HF repos — `HF_USER` is
hardcoded to `"ed-donner"` throughout, with a comment saying "your HF name here!". If you
personally ran this during the course and pushed under *your own* HF username, look for a
repo named `<your-hf-username>/price-<timestamp>` in your HF account — that's your actual
trained adapter, and it's the one that would match your CV claim. If you never actually got
that far (or your account has no such repo), the two `ed-donner/...` repos above are still
useful as a reference/reproduction target, but they aren't weights *you* trained.

**Week 7 result — the key finding of the whole project:**

| Model | Error |
|---|---|
| Base Llama 3.2 3B, 4-bit, no fine-tuning | $110.72 (worse than guessing the mean!) |
| Fine-tuned on 20K "lite" dataset | $65.40 |
| **Fine-tuned on full 800K dataset** | **$39.85** |

A 3-billion-parameter open-weight model, fine-tuned with QLoRA on your own data, beats every
frontier proprietary model tried zero-shot (GPT-5.1's $44.74, Claude 4.5 Sonnet's $47.10) —
at a fraction of the inference cost. This result is what motivates week8: deploy *this* model
as the production "specialist."

---

## 4. Week 8 — Deploy as an autonomous agent system

### Serving the fine-tuned model (Modal.com)
The week7 Llama + LoRA adapter is deployed as a serverless GPU function on **Modal**:
- 4-bit NF4 quantization (`BitsAndBytesConfig`), base model + `PeftModel` LoRA adapter loaded
  from a pinned HuggingFace revision, running on a **T4 GPU**.
- `pricer_service.py` — a stateless version (reloads the model every call).
- `pricer_service2.py` — the production version: an `@app.cls` that loads the model **once
  per container** (`@modal.enter()`), with an `hf-hub-cache` Modal Volume so weights persist
  across cold starts instead of re-downloading.
- Inference: build the same `"What does this cost...? / <description> / Price is $"` prompt,
  generate 5 tokens, regex out the dollar amount.

### The agents (`week8/agents/`)
| Agent | Job | Backing tech |
|---|---|---|
| `SpecialistAgent` | Calls the fine-tuned Llama | Modal-hosted `Pricer` service |
| `FrontierAgent` | RAG-augmented price estimate | Chroma vector DB (800K items, `all-MiniLM-L6-v2` embeddings) + GPT-5.1 |
| `NeuralNetworkAgent` | Inference-only version of the week6 residual DNN | Pre-trained weights, no retraining |
| `EnsembleAgent` | Combines the three above | Weighted average: **0.8×Frontier + 0.1×Specialist + 0.1×NeuralNetwork** |
| `ScannerAgent` | Finds candidate deals | Scrapes 3 dealnews.com RSS feeds + BeautifulSoup, then GPT-5-mini with structured outputs picks the 5 clearest deals |
| `MessagingAgent` | Sends notifications | Pushover API, with Claude Sonnet 4.5 writing the alert copy |
| `PlanningAgent` | Fixed pipeline: scan → price every deal via Ensemble → sort by (estimate − price) discount → alert if discount > $50 | Deterministic orchestration |
| `AutonomousPlanningAgent` | Same job, but lets GPT-5.1 choose the tool-call sequence itself | OpenAI function/tool calling |

### Tying it together
- `deal_agent_framework.py`'s `DealAgentFramework` holds the Chroma client, lazily builds a
  `PlanningAgent`, and persists every surfaced deal (with its estimate and discount) to
  `memory.json` so the app never re-alerts on the same deal twice.
- `price_is_right.py` is the final Gradio app: a live-updating table of deals found so far, a
  3D t-SNE plot of the product embedding space, and a scrolling colorized log of agent
  activity. A `gr.Timer` re-runs the whole scan → price → notify pipeline **every 5 minutes**,
  making it fully autonomous — no user interaction needed once it's running.

**Final leaderboard (`week8/results.ipynb`):**

| Model | Error |
|---|---|
| Fine-tuned Llama (full dataset) | $39.85 |
| GPT-5.1 + RAG | $30.19 |
| **Ensemble (Specialist + Frontier + NeuralNetwork)** | **$29.90 — best overall** |

---

## 5. End-to-end prediction path (as deployed)

1. `ScannerAgent` pulls new listings from RSS feeds, has GPT-5-mini pick out the 5 clearest
   deals with a real price attached.
2. For each deal, `EnsembleAgent`:
   - rewrites the raw scraped text into a clean description (`Preprocessor`),
   - gets a RAG-based estimate from `FrontierAgent` (nearest-neighbor lookup in the 800K-item
     Chroma index + GPT-5.1),
   - gets an estimate from `SpecialistAgent` (the fine-tuned Llama running on Modal),
   - gets an estimate from `NeuralNetworkAgent` (the residual DNN),
   - blends them 0.8 / 0.1 / 0.1.
3. `PlanningAgent` computes `discount = estimate − actual_price` for every deal, keeps the
   biggest one, and if the discount exceeds $50, calls `MessagingAgent` to push a phone
   notification.
4. The result is logged to `memory.json` and shown live in the Gradio dashboard.

---

## Repo layout reference

```
week6/
  day1.ipynb .. day5.ipynb      curation -> preprocessing -> baselines -> DL/frontier zero-shot -> OpenAI fine-tune
  redemption_train/run.ipynb    the big residual deep neural network
  results.ipynb                 week6 leaderboard chart
  pricer/                       items.py, loaders.py, parser.py, preprocessor.py,
                                 batch.py, deep_neural_network.py, evaluator.py
week7/
  day1.ipynb .. day5.ipynb      links to Colab for QLoRA training; day2 does local prompt-prep
  pricer/                       items.py (prompt/completion building), evaluator.py
  util.py                       evaluator for raw prompt/completion dict format
  results.ipynb                 week7 leaderboard chart
week8/
  day1.ipynb .. day5.ipynb      Modal deployment -> RAG/ensemble -> scanner/messaging -> planning -> Gradio app
  agents/                       agent.py, specialist_agent.py, frontier_agent.py,
                                 neural_network_agent.py, ensemble_agent.py, scanner_agent.py,
                                 messaging_agent.py, planning_agent.py, autonomous_planning_agent.py,
                                 deals.py, preprocessor.py, deep_neural_network.py
  deal_agent_framework.py       orchestration + memory.json persistence
  price_is_right.py             the Gradio app (final product)
  hello.py / llama.py / pricer_service*.py / pricer_ephemeral.py   Modal deployment scripts
  memory.json                   persisted record of every deal ever surfaced
  results.ipynb                 final leaderboard chart
```
