# Session 10 — apply the visual design (Tue Nov 3). Approved by Simon.
Reference mockups (static, made with Claude's design canvas): docs/design/landing.mock.html,
docs/design/app.mock.html, docs/design/queue.mock.html. They are DESIGN REFERENCES, not code
to ship: they use a special <x-dc> format; copy the look (layout, colours, type, spacing,
copy), not the markup. The user is present. After each step: tests green -> BUILD_LOG line ->
commit -> push. DO NOT change any logic, endpoint, ID used by JS, or test behaviour.

## Design tokens (put in web/app.css :root, light + dark via prefers-color-scheme)
bg #F7F7F4 · surface #FFFFFF · ink #18201F · muted #4E5B59 / #5E6B69 · line #DDE2DF ·
accent agar #C8892B (text on accent: #18201F) · accent-dark text #7A4E10 on #F6E7CC ·
teal (SAM / verified) #1F6F6A, overlay stroke #29B3A8 · ok #2F7D4F on #DCEFE2 (#1E5E37 text) ·
warn #B7791F · fail #A94A3C. Badges: model #4B3A86 on #E6E0F5, rule #3D4A48 on #E3E8E6,
human #154F4B on #D6ECEA. Fonts: Bricolage Grotesque (headings, big numbers),
Source Sans 3 (body), JetBrains Mono (labels, ids, numbers in tables). Radius 14px cards,
8-10px buttons, 999px pills. Dark mode: derive equivalent tokens, check contrast.

## Step 1 — landing (web/landing.html) to match landing.mock.html
Header with logo mark + nav + "Open the app"; hero 2 columns ("AI counts. Technicians decide.",
CTA "Try an example plate" -> /app?example=14684, "View on GitHub"), plate card on the right
(use a real cached demo overlay image if present in docs/img, else the SVG plate from the mock);
dark band "Counts decide whether a batch is released."; How it works (3 cards); Results with the
two bars; Built with Meta + What it won't do; footer. Keep the "Where it fits" and queue
sections that exist today, restyled.

## Step 2 — app (web/index.html + app.css) to match app.mock.html
- Top bar: logo, nav Analyse / Review queue (with needs-review count from /api/queue) / About.
- Left column: "1 · Plate" card (thumb, name, size, Upload, Camera, examples), "2 · Request"
  card (input + teal mic button + chips; experimental chip dashed amber + label), Run button,
  then "How it was verified" ALWAYS VISIBLE (the live steps list): icon, bold step name,
  values, and the model/rule/human badge with the colours above.
- Right column: result header (big count in Bricolage 72px, the request as subtitle,
  "N by SAM 3.1 + M added by technician" when there are human edits, status pill, Approve);
  viewer card with a segmented control View / Add · draw a box / Remove and the Masks/Boxes
  toggles; under it three cards: Sizes, Report, Export (CSV, Audit log).
- Overlay colours: SAM masks teal (#29B3A8 stroke, 35% fill); human-added masks amber.
- Keep every existing element ID that app.js uses; move them, don't rename them.

## Step 3 — queue (web/queue.html) to match queue.mock.html
Title "Today's plates" + subtitle; reviewer input; "Approve N clean plates"; Export CSV;
4 summary cards (received / need review / clean / approved); filter pills with counts;
table: Plate, Location, Time, CFU, Status pill, "Why it needs you" chips in plain words,
Open / Approve; flagged rows with a warm background. Keep textContent (no innerHTML with data).

## Step 4 — check
Phone width (390 px): columns stack, table scrolls in its box. Dark mode readable.
Update docs/img screenshots for README. 3-line summary in BUILD_LOG.md.
