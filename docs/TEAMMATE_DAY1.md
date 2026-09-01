# BreakRank — teammate Day 1 checklist

*Forward this whole file to your teammate. It is written to be read by them, not by you.*

---

Hi. You own the web stack on this project: frontend, backend, database,
deployment. Vaibhav owns the machine learning. The rule for the next eight
months is simple — **you never write model code, Vaibhav never writes React.**
Every time that line gets crossed, both of you go slower.

Today is setup. It is about four hours of work and none of it is hard, but
everything Vaibhav builds for the next two months lands in the database and the API
you create today, so getting it right now saves days later.

One change from the original plan, and it matters: **we are deploying in week
1–2, not week 6.** The project started two weeks late and demo day (30 October)
cannot move because of Diwali. The way to buy that time back is to get an ugly,
fake, *live* site up now, so that October is spent on the model rather than on
fighting deployment. Read step 5.

---

## 1. Accounts (30 min)

Create all of these. All free, no credit card.

- [ ] **GitHub** — you probably have one. Make sure 2FA is on.
- [ ] **Neon** (neon.tech) — free Postgres. Sign in with GitHub.
- [ ] **Hugging Face** (huggingface.co) — this is where the API will live.
- [ ] **Vercel** (vercel.com) — sign in with GitHub. This is where the site will live.

**Before you create anything, check these three names are free** and tell
Vaibhav immediately if any are taken:

- `github.com/<vaibhavs-username>/breakrank`
- `pypi.org/project/breakrank`
- `breakrank.vercel.app`

Two minutes now saves a rename in October.

## 2. The repository (30 min)

Vaibhav pushes the skeleton first. Once it is up:

- [ ] Accept your collaborator invite
- [ ] `git clone` it
- [ ] Confirm repo settings on github.com:
  - Description: `Ranks Python dependency breaking changes by real-world impact using ML`
  - Topics: `machine-learning`, `python`, `static-analysis`, `developer-tools`, `dependency-management`, `lightgbm`, `learning-to-rank`
  - Website: leave blank, you will add the Vercel URL this week
- [ ] Pin it to your own GitHub profile too — it is your project as much as it is Vaibhav's

**Repo is public from day one.** GitHub Actions minutes are effectively
unlimited on public repos, recruiters can browse without an invite, and a repo
made public in October with one giant "initial commit" looks exactly like what
it is.

## 3. Neon Postgres (30 min)

- [ ] Create a project called `breakrank`
- [ ] Copy the connection string — it looks like
      `postgresql://user:pass@ep-xxx.region.aws.neon.tech/dbname?sslmode=require`
- [ ] Send it to Vaibhav **on a private channel, not in the repo, not in a
      commit, not in a Slack channel with other people in it**
- [ ] Add it as a GitHub Actions secret: repo → Settings → Secrets and
      variables → Actions → New repository secret → name it `DATABASE_URL`

**Never commit a secret. Not once, not "temporarily."** Git history is
permanent and bots scan public GitHub for leaked credentials within minutes
of a push. If you leak one, rotate it at the source immediately — regenerate
the Neon password. Deleting the commit is not enough; the value is already out.

Tables come in week 2, after the API contract is agreed. Do not guess at a
schema today.

## 4. Empty FastAPI app (45 min)

In `api/`. One endpoint, nothing clever:

```python
# api/main.py
from fastapi import FastAPI

app = FastAPI(title="BreakRank API")

@app.get("/health")
def health():
    return {"ok": True, "model_version": "v0-fake"}
```

```
# api/requirements.txt
fastapi
uvicorn[standard]
```

Run it locally with `uvicorn main:app --reload --port 7860` and confirm
`http://localhost:7860/health` returns the JSON. Port 7860 is not arbitrary —
that is what Hugging Face Spaces expects.

## 5. Empty Next.js app, and get it LIVE (90 min) ⭐

This is the important one.

```bash
npx create-next-app@latest web --typescript --tailwind --app
```

Make the homepage say "BreakRank" and show one hardcoded fake number.
Genuinely — hardcoded. Then:

- [ ] Deploy `web/` to Vercel (import the GitHub repo, set root directory to `web`)
- [ ] Deploy `api/` to Hugging Face Spaces (Docker SDK, port 7860)
- [ ] Make the Next.js page fetch `/health` from the HF Space and display the result
- [ ] Send Vaibhav the live URL

**Ship the pipe before the payload.** Most student projects die because
deployment is left for last. If there is a public HTTPS URL that loads by
Friday — even one showing invented numbers — the hardest infrastructure risk
in this project is gone by week 1 instead of week 6, and that is exactly the
two weeks we need to make up.

If Hugging Face Spaces gives you trouble today, deploy just the Vercel
frontend with hardcoded data and come back to the API tomorrow. Do not let the
backend block the URL existing.

## 6. Before you stop (15 min)

- [ ] Commit and push everything on a branch called `web/day-1-setup`, open a
      pull request, let Vaibhav review it, then merge

Yes, a pull request, with only two of you. A recruiter clicking
"Pull requests → Closed" and seeing forty PRs with real descriptions sees a
team that works. Sixty commits straight to `main` saying "update" looks like
a student project. It costs you thirty seconds per change.

Commit messages: `feat(web): ...`, `fix(api): ...`, `chore(deps): ...`.

- [ ] Put the Sunday call in your calendar. 30 minutes, every week, until April.

---

## What is coming for you

| Week | You build |
|---|---|
| 1 | Repo, Neon, empty FastAPI, empty Next.js, **live URL** |
| 2 | Database tables, CI, **API contract agreed and frozen** |
| 3 | API endpoint reading real changes from the DB |
| 4 | A plain page showing that data. Ugly is fine. |
| 5–6 | Real deployment hardening, cron jobs, keep-alive |
| 7–8 | The `requirements.txt` upload flow, rendering results |
| 9–11 | UI polish, "why did this rank high?" panel, mobile, OG image |

**The API contract gets agreed in week 2 and then frozen.** Changing it later
costs a day every time. Vaibhav has the draft — argue about it now, not in
October.

One last thing: **you will be asked "so how does the model work?" in your own
interviews.** Not just Vaibhav. Make them explain it to you properly, more than
once, before October.
