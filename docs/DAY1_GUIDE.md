# BreakRank — Day 1

**Tuesday 1 September 2026 · macOS · absolute beginner · ~6 hours**

---

## Read this bit first

Today you write almost no code. That is deliberate and it is not a waste of a day.

Today you build the thing that lets you build the thing: accounts, a working
Python environment, a public repository, and one script that proves the whole
idea is real. Most final-year projects die in week 9 because of something that
should have been sorted on day 1 — a laptop that cannot install LightGBM, a
repo made public in October with one giant commit, a database URL leaked into
git history. You are going to spend six hours making all of that impossible.

**The one thing that must be true tonight:** a CSV on your laptop containing
real breaking changes from real Python libraries, and a public GitHub repo with
your first commits in it.

**About the schedule.** Your book assumed Day 1 was 17 August. You are starting
two weeks later, and Diwali does not move, so demo day (30 October) does not
move either. The revised dates are in `docs/DATES.md` in your starter files —
read that page today, and read the warning at the bottom of it carefully. The
short version: your teammate deploys the live site in week 1 instead of week 6,
and that is how you get the two weeks back.

---

## Block 0 — Read, don't type (30 min)

Open `BreakRankProjectBook.docx` and read, in this order:

1. **Section B1** — "Rules you must not break", all ten. They take four minutes.
   Rules 4, 5 and 10 are the ones that will save you.
2. **Part 1** — what you are building and why.
3. **Part 2** — the phases and the two gates.

Do not read Parts 4, 5 or 6 yet. Part 4 is for weeks 1–2, alongside building.
Trying to absorb it today will just make you feel behind.

While it is fresh, write these three lines on paper and stick them above your desk:

> Never report accuracy. Report PR-AUC and precision@10.
> Never use a random train/test split. Split by date.
> griffe finds the changes. I rank them. Never blur that line.

You will be asked about all three in interviews.

---

## Block 1 — Names and accounts (45 min)

### 1.1 Check the name is free (5 min)

Before you create anything, open these three in a browser. You want all three
to be **404 / available**:

- `https://github.com/YOURUSERNAME/breakrank`
- `https://pypi.org/project/breakrank`
- `https://breakrank.vercel.app`

If any is taken, pick a variant now (`breakrank-py`, `breakrank-dev`) and use it
everywhere. Two minutes here saves a rename in October.

### 1.2 Create four accounts (40 min)

All free, no credit card, no trial.

| Account | Where | What it is for |
|---|---|---|
| **GitHub** | github.com | Your repo. Turn on 2FA. |
| **Neon** | neon.tech | Free Postgres database. Sign in with GitHub. |
| **Hugging Face** | huggingface.co | Where your API will live from week 5. |
| **Vercel** | vercel.com | Where your website will live. Sign in with GitHub. |

> **"Could I use Firebase instead of Neon?"**
> No — and the reason is worth knowing, because this is exactly the kind of
> trade-off an interviewer asks you to justify.
>
> - **Firestore** (Firebase's free database) allows **20,000 document writes per
>   day**. Your 27 September milestone *alone* is 20,000 labelled rows, and the
>   nightly job rewrites them every night. You would hit the ceiling in week 4
>   and stay there. Reads are capped at 50,000/day too, and one `/analyze`
>   request costs a few thousand — so about twenty demos a day before the site
>   stops working.
> - **Firestore has no joins.** Your labelling step *is* a join: candidate
>   changes matched against the usage index on symbol name. In Firestore you
>   would pull both collections into pandas and join there — at which point the
>   database is doing nothing a CSV was not already doing.
> - **Firebase Data Connect** *is* real Postgres, but the Cloud SQL instance
>   underneath is not free: a 90-day Spark trial (~8,000 ops/day), or a 3-month
>   Blaze trial and then ~$9.37/month. Your project runs to April 2027. Either
>   trial expires around your college review.
>
> Neon is free permanently, 0.5 GB, 100 compute-hours/month, no credit card, and
> speaks ordinary Postgres — so SQLAlchemy works, SQL joins work, and *"I
> designed a normalised schema and wrote the join that generates my labels"* is
> a far better interview answer than *"I used Firestore"*.
>
> Firebase Auth and Firebase Hosting are genuinely good products — you just need
> neither. There is no login in this project, and Vercel beats Firebase Hosting
> for Next.js.

### 1.3 A GitHub token (10 min — and it is optional today)

**Read this first: there is nothing to sign up for.** "Developer settings" is
not a separate product or a developer programme you register for — it is just a
menu item that every GitHub account already has. The reason nobody finds it is
that it sits at the **very bottom** of a long left-hand sidebar, below
Applications, and most people stop scrolling before they reach it.

Skip the hunt entirely — paste this straight into your browser:

> **https://github.com/settings/personal-access-tokens**

If that page loads, you have it. If it asks you to verify your email address
first, do that and come back.

**Create the token:**

1. Click **Generate new token**
2. **Token name:** `breakrank`
3. **Expiration:** 90 days
4. **Repository access:** choose **Public Repositories** — this is the one
   setting that matters
5. Leave every permission dropdown on **No access**. You are only reading
   public data, so it needs no permissions at all.
6. Click **Generate token** at the bottom

The token appears **once**, on a green banner, starting `github_pat_`. Copy it
now — you paste it into a file in Block 3. If you lose it, delete it and make
another one; that costs you two minutes and nothing else.

> The older style of token lives at **https://github.com/settings/tokens**
> ("Tokens (classic)"). It still works and is not deprecated, but GitHub
> recommends fine-grained tokens now, and the fine-grained screen is genuinely
> simpler — one dropdown instead of thirty scary checkboxes. Use the link above.

**If this is fighting you, move on.** Nothing in Day 1 needs this token — every
script you run today talks to PyPI, not to GitHub's API. You need it in **week
5**, when you start pulling changelogs. Without a token GitHub gives you 60 API
requests an hour; with one, 5,000. Leave `GITHUB_TOKEN=` blank in your `.env`
for now and come back to it. Do not lose twenty minutes of Day 1 here.

> Neon and Hugging Face are your teammate's jobs, strictly speaking — but make
> the accounts anyway. You will need them and it takes five minutes.

---

## Block 2 — Get your Mac ready (70 min)

Everything below happens in **Terminal**. Press `⌘ + Space`, type `Terminal`,
press Enter. A window with a text prompt appears. That is where you type
commands. After each command, press Enter and wait for the prompt to come back
before typing the next one.

You will not break your Mac. Nothing here is destructive.

### 2.1 Apple's developer tools (10 min, mostly waiting)

```bash
xcode-select --install
```

A dialog box appears — click **Install**, agree, wait. This gives you `git` and
the compilers Python packages need. If it says *"command line tools are already
installed"*, you are done, move on.

> **This is not Xcode, and it is not a code editor.** "Xcode Command Line
> Tools" is a ~1 GB bundle of compilers plus `git` — no window, no icon, nothing
> to open. It is not the 15 GB Xcode IDE from the App Store, and you do not need
> that. You install this whatever editor you use, because `pip` needs a C
> compiler to build some packages. Your editor is a separate choice — see 2.4.

### 2.2 Homebrew (15 min)

**Check first — it may already be there:**

```bash
brew --version
```

If that prints a version number, skip straight to 2.3. If it says
`command not found`, carry on.

Homebrew installs the software macOS does not ship. Paste this whole line:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It asks for your Mac password (nothing shows as you type — that is normal) and
takes about ten minutes.

**At the end it prints two or three commands under "Next steps".** Copy and run
them. This is the step people skip, and then nothing works. On Apple Silicon
they look like:

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Check it worked:

```bash
brew --version
```

If that prints a version number, you are good. If it says `command not found`,
you missed the "Next steps" commands — scroll up in your terminal and run them.

### 2.3 Python, libomp, and the GitHub CLI (15 min)

```bash
brew install python@3.12 libomp gh
```

Three things:

- **python@3.12** — the macOS system Python is old and you should not touch it.
- **libomp** — Apple's OpenMP runtime. **LightGBM will not import without it.**
  This single missing package is the most common way this project stalls on a
  Mac in week 4. Installing it now costs you nothing.
- **gh** — GitHub's command-line tool. Still install it even if you have already
  created your repo on github.com: it handles your git login in 2.6, and from
  week 1 onwards `gh pr create` is how you open the pull requests that make your
  repo look like a team project rather than a folder of commits.

Confirm:

```bash
python3.12 --version     # should print Python 3.12.x
```

### 2.4 A code editor (10 min)

```bash
brew install --cask visual-studio-code
```

Then open VS Code once, press `⌘ + Shift + P`, type `shell command`, and pick
**"Install 'code' command in PATH"**. Now `code .` opens the current folder from
your terminal, which you will use constantly.

Install two extensions (the squares icon in the left bar): **Python** and
**Ruff**, both by Microsoft/Astral. Ruff will underline unused imports and typos
before you run anything.

**Why VS Code specifically, and not Cursor or Antigravity?** Not because AI
editors are bad — they are good, and Antigravity is a VS Code fork, so every
keybinding and extension you learn transfers to it at zero cost whenever you
want to switch. Three narrower reasons:

- Every tutorial, error message and Stack Overflow answer you hit in the next
  eight months assumes VS Code. Right now you want fewer novel things, not more.
- Antigravity is still a free preview and is openly described as maturing, with
  occasional instability. You have 26 days to a kill date. Do not add a variable
  you cannot debug.
- The real one: **this project's value is that you can explain it.** Part 11 of
  your book is nine pages of interview questions about your own code. An
  agent-first editor is very good at producing code you did not write and cannot
  defend, and that is the specific failure mode your book warns about twice
  (rule 6, rule 7).

Where to use AI freely, starting today — this is a line worth drawing on purpose:

| Use it without hesitation | Write it yourself |
|---|---|
| Explaining a traceback you do not understand | Everything in `ml/` |
| Homebrew / venv / install errors | Your features, labels, and metrics |
| "What does `ast.NodeVisitor` do?" | Anything you would have to defend |
| Boilerplate in `web/` (teammate's side) | Your evaluation code, especially |

The pipeline in `ml/` **is** your placement portfolio — 75% of the weighting in
your own book. Type it yourself, even when it is slower. Everything else, get
all the help you can.

### 2.5 Tell git who you are (5 min)

```bash
git config --global user.name "Vaibhav Panchal"
git config --global user.email "YOUR-GITHUB-EMAIL@example.com"
git config --global init.defaultBranch main
```

**The email must be one that is verified on your GitHub account.** If it is not,
your commits show up as an anonymous grey avatar and **do not count towards your
contribution graph** — and that green stripe from September to April is one of
the first things a recruiter looks at. Eight months of work, invisible.

Check which addresses are on your account at
**https://github.com/settings/emails**, and use one of those exactly.

If you would rather not put your personal address in every public commit, GitHub
gives you a free no-reply alias on that same page — it looks like
`12345678+username@users.noreply.github.com`. It still counts towards your
graph. Use that instead if you prefer.

Verify it took:

```bash
git config --global --list | grep user
```

### 2.6 Log in to GitHub from the terminal (10 min)

```bash
gh auth login
```

Answer: **GitHub.com** → **HTTPS** → **Yes** (authenticate git) → **Login with
a web browser**. It shows an eight-character code, you press Enter, a browser
opens, you paste the code, done.

---

## Block 3 — Create the project (30 min)

### 3.1 Put the starter files in place (5 min)

I have given you a folder called `breakrank`. Move it somewhere sensible and go
into it:

```bash
mv ~/Downloads/breakrank ~/Projects/breakrank
cd ~/Projects/breakrank
ls -a
```

If `~/Projects` does not exist yet: `mkdir -p ~/Projects` first.

`ls -a` should show `ml`, `api`, `web`, `scripts`, `docs`, `README.md`,
`.gitignore`, `LICENSE`, `requirements.txt`, `.env.example`.

**Every command from here on assumes you are inside this folder.** If you open a
new terminal window later, `cd ~/Projects/breakrank` first.

### 3.2 The virtual environment (10 min)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Your prompt now starts with `(.venv)`. That means every `pip install` goes into
this project's own folder rather than being dumped into your system Python — so
this project can use griffe 2.2 while some other project uses griffe 1.x, and
neither breaks the other.

**You must run `source .venv/bin/activate` every time you open a new terminal.**
When you forget — and you will — commands fail with `ModuleNotFoundError`. The
missing `(.venv)` in your prompt is the tell.

### 3.3 Install the libraries (15 min)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Several minutes of scrolling text. Warnings in yellow are fine. Red `ERROR`
lines are not — if you see one, read the last five lines, that is where the real
message is.

### 3.4 Your secrets file (5 min)

```bash
cp .env.example .env
code .env
```

VS Code opens the file. Paste your GitHub token after `GITHUB_TOKEN=` if you made one in
Block 1.3 — if you skipped it, leave the line blank, nothing today reads it. Leave the others
blank for now — your teammate fills in `DATABASE_URL` when Neon is set up. Save
and close.

`.env` is already in `.gitignore`, so it can never be committed. **Do not
remove that line, and do not put a secret anywhere else — not in a comment, not
"temporarily".** Git history is permanent, and bots find leaked credentials in
public repos within minutes.

---

## Block 4 — Check it all actually works (20 min)

```bash
python scripts/day1_check.py
```

This runs seven checks and tells you in plain English what is wrong and how to
fix it. You want every line to say `OK`.

The two that most often fail on a Mac:

- **`Running inside the .venv` fails** → you forgot `source .venv/bin/activate`.
- **`LightGBM imports` fails** → `brew install libomp`, then
  `pip install --force-reinstall lightgbm`.

Do not move on until all seven pass. Everything after this depends on them.

---

## Block 5 — Make it public (45 min)

### 5.1 First commit (15 min)

```bash
git init -b main
git add .
git status
```

**Read the `git status` output before committing.** You should see your source
files. You should *not* see `.env`, `.venv/`, or anything under `data/`. If you
do, stop — `.gitignore` is not being picked up, and it is much easier to fix now
than after it is in history.

```bash
git commit -m "chore: project skeleton, gitignore, licence, dependencies"
```

### 5.2 Push it to GitHub (15 min)

**Which situation are you in?**

---

#### Path A — you have NOT made the repo yet

One command does everything:

```bash
gh repo create breakrank --public --source=. --push \
  --description "Ranks Python dependency breaking changes by real-world impact using ML"
```

---

#### Path B — you already created the repo on github.com

Do not run `gh repo create` — it will fail, because the name is taken by your
own repo. You connect to the one you have instead.

**First, check whether it is empty.** Open it in a browser:

- If you see a big grey box saying **"Quick setup — if you've done this kind of
  thing before"**, it is empty. → **B1**
- If you see a **file list** (a README, maybe a LICENSE or .gitignore), GitHub
  put a starting commit in it. → **B2**

**B1 — empty repo.** Straightforward:

```bash
git remote add origin https://github.com/YOUR-USERNAME/breakrank.git
git push -u origin main
```

**B2 — repo already has files.** Those files are GitHub's boilerplate: an empty
README, a stock .gitignore, a licence. Your local versions of all three are
better and there is nothing on the remote worth keeping. So overwrite it:

```bash
git remote add origin https://github.com/YOUR-USERNAME/breakrank.git
git push -u --force origin main
```

> **On `--force`.** It replaces the remote's history with yours, and it is
> normally something to be very careful with — on a shared repo it can erase a
> teammate's work permanently. It is safe *here*, and only here, because the only
> thing on that remote is an auto-generated commit made by GitHub minutes ago
> that you have not built anything on top of. Once your teammate has cloned this
> repo, do not use `--force` again without talking to them first.

**Then check the repo is public.** Settings → scroll to the bottom → "Change
visibility". If you made it private, switch it to public now.

---

**Public from day one.** Three reasons: GitHub Actions minutes are effectively
unlimited on public repos, recruiters can browse without an invite, and a repo
made public in October with one 4,000-line "initial commit" looks exactly like
what it is.

Confirm it worked either way:

```bash
git remote -v          # should print your repo URL twice
git log --oneline      # should show your commit
```

### 5.3 Dress the window (15 min)

On github.com, in your new repo:

- **About** (gear icon, top right) → add topics:
  `machine-learning`, `python`, `static-analysis`, `developer-tools`,
  `dependency-management`, `lightgbm`, `learning-to-rank`
- **Pin the repo** to your GitHub profile
- **Settings → Collaborators** → invite your teammate
- Send them `docs/TEAMMATE_DAY1.md`

Topics matter more than you would think. They are how a recruiter searching
GitHub for "learning-to-rank" finds work like yours.

---

## Block 6 — The part that is actually the project (60 min)

### 6.1 Get the package list (10 min)

```bash
python ml/ingest/packages.py
```

You get the 2,000 most-downloaded Python packages by rank. Around 3 billion
downloads for `boto3` at number one.

Look at that list for a minute. **Those are the packages you are going to
analyse, and the count next to each one is roughly how many people you would
inconvenience if it broke.** The top 300 get every version downloaded and
diffed. The top 1,500 get their source scanned to see what they actually use.

> Notice the script tries two URLs and caches the result to `data/`. That is not
> me being fussy — Part B2 of your book warns that "JSON field names in public
> APIs change". Verified working on 1 September 2026, and the script prints which
> URL it used so you will know immediately if that stops being true.

### 6.2 Download one package (10 min)

```bash
python ml/ingest/download.py click
```

It lists the last 10 real releases of `click`, downloads the newest source
archive, unpacks it, and tells you how many Python files are inside.

That is the raw material. There is no API for "what changed in this library" —
you have to fetch the source and work it out. That is what makes this project's
data free and infinite, and it is the reason the book chose developer tools over
fintech or healthcare.

### 6.3 See a real diff (10 min)

```bash
python ml/ingest/api_extract.py click
```

griffe reads two versions of `click` and lists what changed in a way that could
break somebody. Real changes, in a library with a billion downloads.

### 6.4 The day's finish line (30 min)

```bash
python scripts/day1_demo.py
```

Six packages, twelve version pairs, downloading and diffing. Takes about two
minutes. When it finishes:

```bash
open data/day1_changes.csv
```

**Sit with this for ten minutes. Genuinely. Do not skip this bit.**

Here is what mine looked like:

```
    230  ( 93%)  ATTRIBUTE_CHANGED_VALUE
     13  (  5%)  OBJECT_REMOVED
      2  (  1%)  PARAMETER_REMOVED
      1  (  0%)  PARAMETER_ADDED_REQUIRED
```

Ninety-three percent of the output is `ATTRIBUTE_CHANGED_VALUE` — a type hint
was rewritten, a constant moved. Almost none of it will break anybody. Buried
underneath, thirteen rows say `OBJECT_REMOVED`: a function that used to exist
and now does not. **Those are the ones that page someone at 3 a.m.**

That is the entire project, visible on your own screen, on day 1. The signal is
real and it is drowning. Everything you build between now and April is about
sorting that list correctly.

You will also see failures scroll past — `attrs` throws an
`AliasResolutionError`, some packages have no source archive at all. **Leave
them.** Week 2 is when you read the failure log and find the one fix that
recovers fifty packages at once. That is real engineering and it is good
interview material; it is not today's problem.

---

## Block 7 — Close the day properly (20 min)

```bash
git add .
git status
```

Check again that `data/` and `.env` are absent. Then:

```bash
git commit -m "feat(ingest): package list, sdist downloader, griffe version diff

Fetches the top 2000 PyPI packages by download count, downloads and
extracts source distributions, and diffs consecutive versions with
griffe to produce candidate breaking changes. Verified end to end on
6 packages: 248 candidate changes, 93% of which are attribute value
changes that are unlikely to break anyone. That ratio is the problem
this project exists to solve."

git push
```

Then:

- [ ] Read `docs/DATES.md` properly, including the warning at the bottom
- [ ] Message your teammate: repo link, `docs/TEAMMATE_DAY1.md`, and the one
      thing that changed — **they deploy in week 1, not week 6**
- [ ] Put a recurring 30-minute Sunday call in both your calendars, now, running
      to April
- [ ] Confirm your actual college review date. If it is not the week of
      30 October, every date in `DATES.md` shifts and you need to know tonight.

---

## Tonight's checklist

- [ ] Four accounts made (GitHub token optional — it is a week-5 problem)
- [ ] `python scripts/day1_check.py` — all seven OK
- [ ] Public GitHub repo, pinned, with topics, teammate invited
- [ ] At least two commits with real messages
- [ ] `data/day1_changes.csv` exists and you have actually looked at it
- [ ] You can say out loud why 93% of those rows do not matter
- [ ] Sunday call in the calendar
- [ ] Teammate has their checklist

Eight boxes. If seven are ticked, that is a good day 1.

---

## What NOT to do today

**Do not start training a model.** You have no labels. There is nothing to
train on. The label comes from the usage index in week 3, and building a model
before then is building on sand.

**Do not tune anything.** No hyperparameters, no feature engineering, no
"let me just try XGBoost as well". Week 4.

**Do not read Part 4 of the book cover to cover.** It is written to be read
*alongside* building, over two weeks. Reading it today will make you feel
behind and teach you nothing that sticks.

**Do not open a Jupyter notebook.** Rule 7: notebooks are for looking at data.
Anything that runs more than once goes in a `.py` file. Starting the pipeline in
a notebook is the single most common way these projects become unmaintainable,
and an interviewer notices within thirty seconds.

**Do not make it pretty.** Not the CSV, not the README, not the terminal
output. Week 11.

---

## Tomorrow (Wednesday 2 September)

Extend `day1_demo.py` from 6 packages to 50. You will hit rate limits, weird
archive layouts, packages with no source distribution, and griffe crashes. Log
every single failure with a reason — do not fix them yet, just record them.

By **Friday 4 September** the downloader should work on any package you name.
By **Sunday 13 September** you need 5,000 rows across 150+ packages.

The kill date is **Sunday 27 September**. Twenty-six days.

---

## When something breaks

In order:

1. **Read the last five lines of the error.** The real message is at the bottom,
   not the top. The top is just the path the error took to get there.
2. **Is `(.venv)` in your prompt?** Half of all `ModuleNotFoundError`s are this.
3. **Are you in `~/Projects/breakrank`?** Run `pwd` and check.
4. **Appendix C of your book** — troubleshooting and error reference.
5. **Ask me.** Paste the whole error, tell me which block you were on.

Nothing you can type today will break your Mac. The worst case is deleting the
`breakrank` folder and starting Block 3 again, which takes twenty minutes.
