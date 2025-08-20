## Text Cleaning for Classification with Fine-tuned T5

---

### 1. Problem Statement

In enterprise settings, many downstream **classification tasks** (e.g., invoice detection, spam filtering, intent classification) require clean textual input.
Using raw enterprise emails or documents directly often introduces **noise** such as:

* Greetings (*Dear, Hi, Best regards…*)
* Signatures (*name, phone, email, LinkedIn, etc.*)
* Automatic footers and disclaimers
* Random characters, separators (`---`)
* HTML/CSS fragments

These noisy tokens cause embeddings to become **overly dispersed**, which in turn **reduces classifier accuracy**.

To address this, instead of relying on **LLMs without probability guarantees**, we fine-tune a **smaller, controllable model (T5-small)** specifically for **text cleaning**.

---

### 2. Approach

We fine-tuned **Google T5-small** (encoder-decoder architecture) on **\~6000 labeled enterprise emails**.

* **Input:** raw email text with greetings, signatures, HTML, etc.
* **Output:** cleaned, semantically meaningful sentence(s).

> Example:
> Input → `"Dear team, please find attached invoice INV-2023-0815 ... Best regards, John Doe +1-202-555-0199 ... <style>...</style>"`
> Output → `"Attached invoice INV-2023-0815 for services provided in July."`

* **Training details:**

  * Max input length = 512 tokens (99% of emails fit within this limit).
  * Model: `t5-small` with gradient checkpointing, fp16, accumulation steps.
  * Dataset is **internal company data** → not public due to confidentiality.

---

### 3. Why T5 for Cleaning?

* **Encoder-decoder** architecture allows flexible **text-to-text transformation**.
* More robust than rule-based regex cleaning (regex cannot remove all noise like greetings, dynamic signatures, or inline CSS).
* Produces **human-readable, semantically consistent output**, not just “filtered text.”

---

### 4. Example

#### Raw Email (Input to T5)

```
Dear accounting team,

I hope you are doing well.
Please find attached the invoice INV-2023-0815 for services provided in July.

If you have any questions, feel free to reach out.

Best regards,
John Doe
Finance Department
Phone: +1-202-555-0199
Email: john.doe@company.com
LinkedIn: linkedin.com/in/johndoe
---
<style>
div { color: red; }
</style>
```

#### Cleaned Email (Output from T5)

```
Attached invoice INV-2023-0815 for services provided in July.
```

---

### 5. Evaluation: Impact on Classification

After cleaning with fine-tuned **T5-small**, we passed the text into **XLM-RoBERTa embeddings** + a classification head.

**Result:**

* **Precision, Recall, and F1-score** improved significantly compared to using raw noisy emails.
* Embeddings became **more compact** and **semantically aligned**, making the classifier more confident and stable.
* Noise reduction directly boosted downstream classification tasks (e.g., invoice detection).

---

### 6. Benefits

* **Reduced noise** → embeddings become more compact and consistent.
* **Improved classification accuracy** → precision, recall, f1-score all increased.
* **Lightweight** → fine-tuned T5-small runs on a single 12GB GPU (RTX 3060).
* **Enterprise-ready** → reproducible, controllable, no dependency on black-box LLM APIs.

---

### 7. Limitations

* Input longer than **512 tokens** must be truncated or split.
* Edge cases (very unusual formats) may still require **post-processing rules**.
* Model trained on **private enterprise data** → cannot be open-sourced directly.

---

With this setup, enterprise classification pipelines gain **clean, reliable inputs** → leading to **higher precision, recall, and f1-score** in downstream tasks using **XLM-RoBERTa + classifier head**.

---
