"""
Streamlit dashboard: analytics, topics, drafts, content queue, alerts.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import desc, select

from config import EXPORTS_DIR, PROJECT_ROOT
from src.alerts.cta_detector import run_cta_detection
from src.analytics.csv_parser import import_exports_folder
from src.db.models import Alert, Draft, Note, Post, Thread, TopicScore
from src.db.session import get_session, init_db
from src.pipeline import (
    generate_draft_for_topic,
    generate_notes_and_threads_for_draft,
    refresh_trends_and_rank,
    run_full_pipeline,
)
from src.publishing.bluesky_publisher import post_thread


def _bootstrap() -> None:
    """Load env and ensure DB exists."""
    load_dotenv(PROJECT_ROOT / ".env")
    st.session_state.setdefault("engine", init_db())


def _pub_url() -> str:
    """Substack publication URL from env."""
    return os.environ.get(
        "SUBSTACK_PUBLICATION_URL",
        "https://greymattersbullock.substack.com",
    ).rstrip("/")


def main() -> None:
    """Render the five-tab dashboard."""
    st.set_page_config(page_title="Grey Matters — bull_sub", layout="wide")
    _bootstrap()
    engine = st.session_state["engine"]
    session = get_session(engine)

    try:
        tab_analytics, tab_topics, tab_drafts, tab_queue, tab_alerts = st.tabs(
            ["Analytics", "Topic ideas", "Drafts", "Content queue", "Alerts"]
        )

        with tab_analytics:
            st.subheader("Performance")

            exports_path = EXPORTS_DIR.resolve()
            EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

            with st.expander("How to get a CSV from Substack", expanded=False):
                st.markdown(
                    """
1. In Substack: **Dashboard → Settings → Exports** (left sidebar).
2. Click **Create new export**. When it is ready, download the **ZIP** from that page (Substack may also email you).
3. Unzip the archive and find CSV file(s) with **post stats** (often something like posts or statistics; column names should include title, dates, and open rates).
4. Either **upload below** or copy the `.csv` file into this folder on your computer:

   `{path}`

5. Click **Import CSV** below. If you see `files=0`, no `.csv` was found in that folder.

**Note:** Substack’s export layout can change. If import shows many **skipped** rows, the CSV may use different column names—open an issue or adjust the file to include `title`, `published_at`, `open_rate`, and `click_rate` (or similar).
""".format(
                        path=exports_path
                    )
                )

            uploaded = st.file_uploader("Upload a Substack stats CSV", type=["csv"], key="substack_csv_upload")
            if uploaded is not None and st.button("Save uploaded file to data/exports", key="save_csv_upload"):
                dest = exports_path / Path(uploaded.name).name
                dest.write_bytes(uploaded.getvalue())
                st.session_state["import_msg"] = f"Saved file to {dest}. Click Import CSV below."
                st.rerun()

            if "import_msg" in st.session_state:
                msg = st.session_state.pop("import_msg")
                if msg.startswith("Import failed") or "failed" in msg.lower():
                    st.error(msg)
                else:
                    st.success(msg)

            if st.button("Import CSV from data/exports", key="import_csv_btn"):
                try:
                    imp = import_exports_folder(session)
                    run_cta_detection(session)
                    session.commit()
                    st.session_state["import_msg"] = (
                        f"Imported: files={imp.files_seen} rows={imp.rows_seen} "
                        f"upserted={imp.upserted_posts} skipped={imp.skipped_rows}. "
                        f"If upserted=0, check the CSV has title, date, and open/click columns."
                    )
                except Exception as e:
                    st.session_state["import_msg"] = f"Import failed: {e}"
                st.rerun()

            posts = session.scalars(select(Post).order_by(desc(Post.published_at))).all()
            if posts:
                df = pd.DataFrame(
                    [
                        {
                            "title": p.title,
                            "published_at": p.published_at,
                            "open_rate": p.open_rate,
                            "click_rate": p.click_rate,
                            "paid_conversions": p.paid_conversions,
                            "topics": p.topics,
                        }
                        for p in posts
                    ]
                )
                last20 = df.sort_values("published_at", ascending=False).head(20)
                chart_df = last20.dropna(subset=["open_rate"]).set_index("published_at")["open_rate"].astype(
                    float
                )
                if not chart_df.empty:
                    st.line_chart(chart_df)

                topics_df = df.dropna(subset=["topics", "open_rate"])
                if not topics_df.empty:
                    explode = topics_df.assign(topic=topics_df["topics"].str.split(r"[;,]")).explode("topic")
                    explode["topic"] = explode["topic"].str.strip()
                    top_open = (
                        explode.groupby("topic")["open_rate"].mean().sort_values(ascending=False).head(10)
                    )
                    if not top_open.empty:
                        st.bar_chart(top_open)

                paid = df.dropna(subset=["topics", "paid_conversions"])
                if not paid.empty:
                    pe = paid.assign(topic=paid["topics"].str.split(r"[;,]")).explode("topic")
                    pe["topic"] = pe["topic"].str.strip()
                    top_paid = (
                        pe.groupby("topic")["paid_conversions"]
                        .sum()
                        .sort_values(ascending=False)
                        .head(10)
                    )
                    if not top_paid.empty:
                        st.bar_chart(top_paid)
                st.dataframe(df.sort_values("published_at", ascending=False), width="stretch")
            else:
                st.info("No posts yet. Import a Substack CSV export (see expander above).")

        with tab_topics:
            st.subheader("Scored topics")
            if st.button("Refresh trends"):
                with st.spinner("Fetching Google Trends…"):
                    refresh_trends_and_rank(session, force_refresh=True)
                    session.commit()
                st.success("Trends refreshed.")
                st.rerun()

            scores = session.scalars(
                select(TopicScore).order_by(desc(TopicScore.fetched_at), desc(TopicScore.combined_score))
            ).all()
            if scores:
                latest_ts = max(s.fetched_at for s in scores)
                latest = [s for s in scores if s.fetched_at == latest_ts]
                tdf = pd.DataFrame(
                    [
                        {
                            "topic": s.topic,
                            "combined_score": s.combined_score,
                            "trend_score": s.trend_score,
                            "historical_score": s.historical_performance_score,
                            "fetched_at": s.fetched_at,
                        }
                        for s in latest
                    ]
                )
                st.dataframe(tdf, width="stretch")

                for _, row in tdf.iterrows():
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.write(f"**{row['topic']}** — combined {row['combined_score']:.1f}")
                    with c2:
                        if st.button("Generate", key=f"gen_{row['topic']}"):
                            with st.spinner("Generating draft…"):
                                d = generate_draft_for_topic(
                                    session,
                                    str(row["topic"]),
                                    combined_score=float(row["combined_score"]),
                                )
                                if d:
                                    session.commit()
                                    st.success(f"Draft created (id={d.id}). See Drafts tab.")
                                else:
                                    st.error("Generation failed (check Gemini / logs).")
                            st.rerun()
            else:
                st.info('Click "Refresh trends" to populate topic scores.')

        with tab_drafts:
            st.subheader("Pending and approved drafts")
            drafts = session.scalars(select(Draft).order_by(desc(Draft.created_at))).all()
            for d in drafts:
                with st.expander(f"{d.status.upper()} · {d.title[:80]} (topic: {d.topic})"):
                    st.markdown(f"**Score:** {d.combined_score}")
                    if d.cover_image_prompt:
                        st.subheader("Cover image prompt (paste into DALL-E, Midjourney, etc.)")
                        st.caption("No image is generated here—copy the text below into your image tool, then upload the result in Substack.")
                        st.code(d.cover_image_prompt, language=None)
                    st.markdown(d.content[:4000] + ("…" if len(d.content) > 4000 else ""))
                    if d.title_variants:
                        st.write("Title variants (scores from your historical patterns):")
                        for v in d.title_variants:
                            rec = " (recommended)" if v.get("recommended") else ""
                            st.write(f"- {v.get('title')} — score {v.get('score')}{rec}")
                        choices = [str(v.get("title", "")) for v in d.title_variants if v.get("title")]
                        if choices:
                            rec_i = 0
                            for v in d.title_variants:
                                if v.get("recommended") and str(v.get("title")) in choices:
                                    rec_i = choices.index(str(v["title"]))
                                    break
                            pick = st.selectbox(
                                "Use this title for the draft",
                                choices,
                                key=f"title_pick_{d.id}",
                                index=rec_i,
                            )
                            if st.button("Apply selected title", key=f"apply_title_{d.id}"):
                                d.title = pick[:500]
                                for v in d.title_variants:
                                    v["recommended"] = v.get("title") == pick
                                session.commit()
                                st.rerun()

                    c1, c2 = st.columns(2)
                    with c1:
                        if d.status == "pending" and st.button("Approve", key=f"app_{d.id}"):
                            d.status = "approved"
                            session.commit()
                            st.rerun()
                    with c2:
                        if d.status == "pending" and st.button("Reject", key=f"rej_{d.id}"):
                            d.status = "rejected"
                            session.commit()
                            st.rerun()

                    if d.status == "approved":
                        url_key = f"article_url_{d.id}"
                        st.text_input(
                            "Article URL for thread CTA (full public URL)",
                            value=f"{_pub_url()}/p/your-slug",
                            key=url_key,
                        )
                        if st.button("Generate notes + threads", key=f"notes_{d.id}"):
                            url = st.session_state.get(url_key, "").strip()
                            if not url:
                                st.error("Set the article URL first.")
                            else:
                                with st.spinner("Generating…"):
                                    generate_notes_and_threads_for_draft(session, d, url)
                                    session.commit()
                                st.success("Notes and threads generated. Open Content queue.")
                                st.rerun()

        with tab_queue:
            st.subheader("Notes (copy from code blocks) and threads")
            notes = session.scalars(select(Note).order_by(desc(Note.created_at))).all()
            for n in notes:
                draft = session.get(Draft, n.draft_id)
                st.markdown(f"### Draft {n.draft_id} — {draft.topic if draft else ''}")
                st.caption(
                    f"Schedule: {n.scheduled_day_1}, {n.scheduled_day_2}, {n.scheduled_day_3} · {n.status}"
                )
                st.caption("Note 1 (hook)")
                st.code(n.note_1, language=None)
                st.caption("Note 2 (personal)")
                st.code(n.note_2, language=None)
                st.caption("Note 3 (question)")
                st.code(n.note_3, language=None)
                st.divider()

            threads = session.scalars(select(Thread).order_by(desc(Thread.created_at))).all()
            for t in threads:
                st.markdown(f"**{t.platform}** · draft {t.draft_id} · **{t.status}**")
                for i, line in enumerate(t.thread_content):
                    st.text(f"{i + 1}. {line}")
                if t.platform == "bluesky" and t.status == "pending":
                    if st.button("Post to Bluesky", key=f"bsky_{t.id}"):
                        uri = post_thread(list(t.thread_content))
                        if uri:
                            t.status = "posted"
                            session.commit()
                            st.success(f"Posted. Root: {uri}")
                        else:
                            st.error("Bluesky post failed (see logs).")
                        st.rerun()

        with tab_alerts:
            st.subheader("Alerts and pipeline")
            if st.button("Run pipeline now"):
                msg = run_full_pipeline()
                st.success(msg)
                st.rerun()

            alerts = session.scalars(select(Alert).order_by(desc(Alert.triggered_at))).all()
            for a in alerts:
                st.markdown(f"**{a.alert_type}** · post_id={a.post_id} · actioned={a.actioned}")
                if a.cta_draft:
                    st.text_area("Draft", a.cta_draft, height=150, key=f"cta_{a.id}")
                if not a.actioned and st.button("Mark actioned", key=f"act_{a.id}"):
                    a.actioned = True
                    session.commit()
                    st.rerun()
    finally:
        session.close()


if __name__ == "__main__":
    import subprocess
    import sys

    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        main()
    else:
        script_path = os.path.abspath(__file__)
        raise SystemExit(
            subprocess.call(
                [sys.executable, "-m", "streamlit", "run", script_path, *sys.argv[1:]],
            )
        )
