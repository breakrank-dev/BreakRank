// Where the API lives, and how to turn an error response into a sentence.
// NEXT_PUBLIC_ variables are baked in when Vercel builds the site, so a
// missing one means "fix the setting and redeploy", not "try again".

export const API = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/+$/, "");

export const API_MISSING_MESSAGE =
  "This site was built without NEXT_PUBLIC_API_URL, so it cannot reach the API. " +
  "Set it in Vercel (Settings → Environment Variables) and redeploy.";

export const API_UNREACHABLE_MESSAGE =
  "Could not reach the API. It sleeps after 15 minutes idle on the free tier — give it a minute and try again.";

// Reads both the current error shape ({"detail": {"message"}}) and the planned
// one ({"error": {"message"}}), so the page keeps working across that change.
export async function errorMessage(res: Response, fallback: string): Promise<string> {
  const body = await res.json().catch(() => null);
  return body?.error?.message ?? body?.detail?.message ?? fallback;
}
