export type MeetKind = "zoom" | "teams";

export type MeetLink = { href: string; kind: MeetKind };

const ZOOM_JOIN = "https://zoom.us/join";
const TEAMS_HOME = "https://teams.microsoft.com";

export function defaultMeetHref(kind: MeetKind): string {
  return kind === "zoom" ? ZOOM_JOIN : TEAMS_HOME;
}

export function parseMeetingLink(kind: MeetKind, raw: string): MeetLink | { error: string } {
  const t = raw.trim();
  if (!t) return { href: defaultMeetHref(kind), kind };
  return kind === "zoom" ? parseZoom(t) : parseTeams(t);
}

function parseZoom(t: string): MeetLink | { error: string } {
  if (/^https?:\/\//i.test(t)) {
    try {
      const u = new URL(t);
      const host = u.hostname.toLowerCase();
      if (host !== "zoom.us" && !host.endsWith(".zoom.us")) {
        return { error: "That does not look like a Zoom link." };
      }
      return { href: u.toString(), kind: "zoom" };
    } catch {
      return { error: "That Zoom link is not valid." };
    }
  }
  const id = t.replace(/[^\d]/g, "");
  if (id.length >= 9 && id.length <= 13) {
    return { href: `https://zoom.us/j/${id}`, kind: "zoom" };
  }
  return { error: "Paste a zoom.us link or a 9-13 digit meeting ID." };
}

function parseTeams(t: string): MeetLink | { error: string } {
  if (/^msteams:/i.test(t)) return { href: t, kind: "teams" };
  if (/^https?:\/\//i.test(t)) {
    try {
      const u = new URL(t);
      const host = u.hostname.toLowerCase();
      const ok =
        host === "teams.microsoft.com" ||
        host.endsWith(".teams.microsoft.com") ||
        host === "teams.live.com" ||
        host.endsWith(".teams.live.com") ||
        host.endsWith(".office.com") ||
        host.endsWith("microsoft.com") ||
        host.endsWith(".cloud.microsoft");
      if (!ok) return { error: "That does not look like a Teams link." };
      return { href: u.toString(), kind: "teams" };
    } catch {
      return { error: "That Teams link is not valid." };
    }
  }
  return { error: "Paste a Teams invite link from the calendar or chat." };
}

export function openMeeting(href: string): void {
  window.open(href, "_blank", "noopener,noreferrer");
}
