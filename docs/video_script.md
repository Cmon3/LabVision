# LabVision Copilot — 3-minute demo script (draft)

Tip: record the screen with Cmd+Shift+5 (QuickTime), voice in one take per section,
then join. Read slowly: ~130 words per minute -> ~390 words total.

## 0:00–0:15 · Hook
[Screen: app with an agar plate]
"Everything you'll see was built this week, during the hackathon.
Microbiology labs still count bacterial colonies by eye. In pharma quality control,
those counts decide whether a batch is released. LabVision Copilot does it from a photo."

## 0:15–1:15 · Demo
[Click the first example plate, type: "count colonies larger than 2 mm", Run]
"I give it a plate and a plain request. Muse Spark Open, Meta's open-weights model,
plans the steps. SAM 3.1 finds every colony. Then plain Python measures them:
the plate is 90 millimetres, so pixels become millimetres."
[Open tab "How it was verified"]
"Every step is visible: image quality check, the prompt SAM used, the plate diameter
it detected. No number in the report comes from the language model."
[Click the dense plate example]
"This plate has about 300 colonies. A technician wouldn't trust a count here, and
neither does the agent: too numerous to count, dilute and re-plate."

## 1:15–2:05 · Does it work?
[Screen: README results / RESULTS.md chart]
"We measured it. On 20 real plates the agent had never seen, the count is off by
11.8% on average, and 7 plates are exact.
The biggest lesson: asking SAM for 'bacterial colony' fails, 99% error.
Asking for 'round spot' works. Visual words beat technical words.
We tuned on one half of the plates and tested on the other, so the number is honest."

## 2:05–2:45 · What it won't do
"We also tried to identify species. Muse declined, for safety reasons, and we kept
that boundary. We tried flagging contaminated plates; it didn't beat the simple
baseline, so a rule we wrote in advance removed it from the product.
This is a verifiable agent: it shows its work and tells you what it can't do."

## 2:30–2:50 · From one plate to a whole session
[Screen: inbox folder → queue filling live, 3 s refresh]
"And it scales. With a phone on a stand or an incubator's camera, photos land in the inbox. The agent runs the fixed pipeline — no prompt guessing. Only flagged plates need a human: Re-photo for blur, TNTC for dense, Empty, or low confidence. You approve with your name, and the audit log goes straight to the LIMS as CSV."

## 2:50–3:00 · Close
"SAM 3.1 on the Meta Model API, Muse Spark Open on Baseten, fully open source.
LabVision Copilot: from plate photo to a count you can audit, in seconds."

---
## X post (draft)
Built LabVision Copilot for the @AIatMeta hackathon 🧫
Plate photo + plain request → colony count you can audit.
• SAM 3.1 finds colonies, Python measures (no LLM numbers)
• 11.8% avg error on held-out plates
• "round spot" beats "bacterial colony" (99% → 12% error)
Demo + code: <link>
