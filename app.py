# ============================
# PART 4/4 — Downloads + Packaging (ZIP) + Script Backup Export
# (append below Part 3 in app.py)
# ============================

def _zip_bytes(file_map: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, b in file_map.items():
            z.writestr(name, b)
    return bio.getvalue()


def _export_scripts_zip() -> bytes:
    proj_slug = _slugify(st.session_state.project_title)

    file_map: Dict[str, bytes] = {}
    intro = (st.session_state.intro_text or "").strip()
    outro = (st.session_state.outro_text or "").strip()

    if intro:
        file_map[f"{proj_slug}__Intro.txt"] = intro.encode("utf-8")

    for idx, ch in enumerate(st.session_state.chapters):
        text = (ch.get("text") or "").strip()
        if text:
            file_map[f"{proj_slug}__Chapter_{idx+1:02d}.txt"] = text.encode("utf-8")

    if outro:
        file_map[f"{proj_slug}__Outro.txt"] = outro.encode("utf-8")

    return _zip_bytes(file_map)


# --- UI: show generated files + downloads in the right column area
# (We can render in-place; Streamlit columns persist across reruns.)
st.markdown("---")
st.subheader("Outputs")

generated = st.session_state.generated_files or {}

if not generated:
    st.caption("No MP3s generated yet.")
else:
    st.caption("Download individual MP3s or download everything as a ZIP.")

    # Individual downloads
    for fname, b in generated.items():
        st.download_button(
            label=f"Download {fname}",
            data=b,
            file_name=fname,
            mime="audio/mpeg",
            use_container_width=True,
            key=f"dl_{fname}",
        )

# ZIP download button (wired to the button in Part 2)
if st.session_state.get("download_zip_btn"):
    st.session_state.download_zip_btn = False
    if not generated:
        st.warning("No MP3s to zip yet. Click **Create Audio** first.")
    else:
        zip_b = _zip_bytes(generated)
        proj_slug = _slugify(st.session_state.project_title)
        st.download_button(
            label="Download ALL MP3s as ZIP",
            data=zip_b,
            file_name=f"{proj_slug}__mp3s.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_all_zip_now",
        )

# Script export button
if st.session_state.get("export_scripts_btn"):
    st.session_state.export_scripts_btn = False
    zip_b = _export_scripts_zip()
    proj_slug = _slugify(st.session_state.project_title)
    st.download_button(
        label="Download Script Backup (ZIP)",
        data=zip_b,
        file_name=f"{proj_slug}__scripts.zip",
        mime="application/zip",
        use_container_width=True,
        key="dl_scripts_zip_now",
    )
