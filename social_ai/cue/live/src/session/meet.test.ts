import assert from "node:assert/strict";
import test from "node:test";
import { parseMeetingLink } from "./meet.ts";

test("empty Zoom field opens the join page", () => {
  const r = parseMeetingLink("zoom", "  ");
  assert.equal("href" in r && r.href, "https://zoom.us/join");
});

test("Zoom meeting ID becomes a join URL", () => {
  const r = parseMeetingLink("zoom", "123 456 7890");
  assert.deepEqual(r, { href: "https://zoom.us/j/1234567890", kind: "zoom" });
});

test("Zoom invite is kept", () => {
  const r = parseMeetingLink("zoom", "https://us02web.zoom.us/j/123456789?pwd=abc");
  assert.equal("href" in r && r.href.includes("us02web.zoom.us/j/123456789"), true);
});

test("non-Zoom URL is rejected", () => {
  const r = parseMeetingLink("zoom", "https://example.com/j/1");
  assert.equal("error" in r, true);
});

test("Teams invite is kept", () => {
  const r = parseMeetingLink(
    "teams",
    "https://teams.microsoft.com/l/meetup-join/19%3ameeting_abc/0?context=%7b%7d",
  );
  assert.equal("href" in r && r.href.includes("teams.microsoft.com"), true);
});

test("bare Teams text is rejected", () => {
  const r = parseMeetingLink("teams", "standup tomorrow");
  assert.equal("error" in r, true);
});
