# The dates card — revised 1 September 2026

Print this. Stick it where you can see it.

The project book was written on 16 August 2026 and assumed Day 1 was
**Monday 17 August**. You are starting **Tuesday 1 September**, so every
milestone below has moved **two weeks later** — except the ones that cannot
move. Read the warning at the bottom before you file this away.

---

## Revised milestones

| Date | What must be true | Slip = |
|---|---|---|
| **Tue 1 Sep 2026** | Day 1. Accounts, repo, environment, first real CSV rows. | — |
| Fri 4 Sep 2026 | Downloader script working on any package you name. | 1 day |
| **Sun 13 Sep 2026** | **5,000 real API-change rows in a CSV.** ≥150 distinct packages. | 2 days |
| 🔴 **Sun 27 Sep 2026** | **KILL DATE.** ≥20,000 labelled rows **AND** a model beating the version-number baseline on PR-AUC. | project |
| 🟡 **Sun 11 Oct 2026** | Website live on the public internet. | 3 days |
| 🟡 **Sun 25 Oct 2026** | Ranking works. Paste a `requirements.txt`, get an ordered list. If the site is not live tonight, stop adding features. | 3 days |
| ⭐ **Fri 30 Oct 2026** | **DEMO READY.** Pre-Diwali review + placement interviews. **This date does not move.** | — |
| Sun 8 Nov 2026 | Diwali. College goes quiet from ~4 Nov. | — |
| Tue 1 Dec 2026 | Freeze the "future test set". | — |
| Sun 20 Dec 2026 | College review build: npm support, GitHub bot, written report. | — |
| April 2027 | Final submission. | — |

---

## ⚠️ The thing you have to decide about

Shifting everything by two weeks *would* push demo day from 30 October to
**13 November** — which is **after Diwali**, after your review, and after the
placement interviews you are building this for.

Diwali does not move. So the demo date does not move either.

That means the two weeks have to be absorbed somewhere. Between the kill date
(27 Sep) and demo day (30 Oct) you now have **33 days** to do what the book
budgeted seven weeks for.

**The fix, and it is a good one: bring deployment forward.**

The book has your teammate deploying the live URL in week 6. Do it in
**week 1–2 instead**. An empty FastAPI app returning fake numbers, deployed to
Hugging Face Spaces, with a Next.js page on Vercel that displays them. Ugly,
fake, live.

This is not a compromise — it is the book's own rule ("ship the pipe before
the payload", Part B1 rule 3) applied earlier. It buys back most of the two
weeks, because the thing that normally eats October is deployment, and it
removes the single biggest risk to the 11 October gate.

Your teammate has the capacity for this. Their week 1 in the book is repo +
database + empty apps, which is half a day of real work. Give them
deployment too. Their Day 1 checklist already reflects this.

**Also worth doing today:** confirm your actual college review date. If it is
not the week of 30 October, every date above shifts by that offset instead,
and this whole page needs rewriting. Two minutes on WhatsApp now.

---

## The two gates, in plain words

**🔴 27 September — the kill date.** Two conditions, both must be true that
night: at least 20,000 labelled rows, and a model that beats the
version-number baseline on PR-AUC. If both are true, continue. If either
fails, open Part 13 the next morning and switch to a backup project.

**After 27 September you do not switch. Not for any reason.** Switching later
leaves under five weeks and you walk into interviews with two unfinished
projects — strictly worse than one finished average one.

**🟡 25 October — the second gate.** If the site is not live and working that
night, stop adding features. Spend everything left on making it work.

---

## Every Sunday

Open Part 5 of the book. Check what should have shipped. Plan the week.
One 30-minute call with your teammate: what shipped, what is blocked,
what is next.
