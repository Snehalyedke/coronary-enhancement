(() => {
  const $ = (id) => document.getElementById(id);

  // ---------- tabs ----------
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("active", x === t));
      document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "panel-" + t.dataset.tab));
    })
  );

  function setStatus(kind, msg, isError = false) {
    const el = $("status-" + kind);
    el.textContent = msg || "";
    el.classList.toggle("error", isError);
  }

  // ---------- generic drop-zone wiring ----------
  function wireDrop(kind, onFile) {
    const drop = $("drop-" + kind), input = $("file-" + kind);
    input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
    ["dragenter", "dragover"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (ev) => {
      const f = ev.dataTransfer.files[0];
      if (f) { try { const dt = new DataTransfer(); dt.items.add(f); input.files = dt.files; } catch (_) {} onFile(f); }
    });
  }

  // =====================================================================  IMAGE
  let imageFile = null;
  wireDrop("image", (f) => {
    imageFile = f;
    $("name-image").textContent = f.name;
    $("go-image").disabled = false;
    $("results-image").hidden = true; $("dl-image").hidden = true;
    setStatus("image", "");
  });

  $("go-image").addEventListener("click", async () => {
    if (!imageFile) return;
    const fd = new FormData();
    fd.append("file", imageFile);
    fd.append("vessels_dark", $("dark-image").checked ? "1" : "0");
    $("go-image").disabled = true;
    setStatus("image", "Enhancing…");
    try {
      const r = await fetch("/api/image", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Request failed");
      const bust = "?t=" + Date.now();
      $("img-original").src = d.original + bust;
      $("img-enhanced").src = d.enhanced + bust;
      $("dl-image-8").href = d.enhanced;
      $("dl-image-16").href = d.enhanced16;
      $("results-image").hidden = false; $("dl-image").hidden = false;
      setStatus("image", `Done – ${d.width}×${d.height}px, processed in ${d.ms} ms.`);
    } catch (e) {
      setStatus("image", e.message, true);
    } finally {
      $("go-image").disabled = false;
    }
  });

  // =====================================================================  VIDEO
  let videoFile = null, localUrl = null;
  wireDrop("video", (f) => {
    videoFile = f;
    $("name-video").textContent = f.name;
    $("go-video").disabled = false;
    $("results-video").hidden = true; $("dl-video").hidden = true; $("progress-video").hidden = true;
    setStatus("video", "");
  });

  function showOriginalPreview(file) {
    // Only works for formats the browser can play (mp4/webm mostly); silently skipped otherwise.
    if (localUrl) URL.revokeObjectURL(localUrl);
    const v = $("vid-original");
    const fig = $("fig-video-original");
    fig.hidden = true;
    if (/\.(dcm|dicom)$/i.test(file.name)) return;
    localUrl = URL.createObjectURL(file);
    v.onloadeddata = () => { fig.hidden = false; v.play().catch(() => {}); };
    v.onerror = () => { fig.hidden = true; };
    v.src = localUrl;
  }

  async function poll(jobId) {
    const bar = $("bar-video");
    while (true) {
      const r = await fetch("/api/jobs/" + jobId);
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || "Job lost");
      if (j.status === "error") throw new Error(j.error);
      if (j.status === "done") return j;
      if (j.total > 0) {
        bar.classList.remove("indeterminate");
        bar.style.width = Math.round((j.done / j.total) * 100) + "%";
        setStatus("video", `Processing frame ${j.done} / ${j.total}…`);
      } else {
        bar.classList.add("indeterminate");
        setStatus("video", j.status === "queued" ? "Queued…" : `Processed ${j.done} frames…`);
      }
      await new Promise((res) => setTimeout(res, 500));
    }
  }

  $("go-video").addEventListener("click", async () => {
    if (!videoFile) return;
    const fd = new FormData();
    fd.append("file", videoFile);
    fd.append("mode", $("mode-video").value);
    fd.append("vessels_dark", $("dark-video").checked ? "1" : "0");
    $("go-video").disabled = true;
    $("results-video").hidden = true; $("dl-video").hidden = true;
    $("progress-video").hidden = false;
    $("bar-video").style.width = "0";
    $("bar-video").classList.add("indeterminate");
    setStatus("video", "Uploading…");
    try {
      const r = await fetch("/api/video", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error || "Upload failed");
      const j = await poll(d.job_id);
      $("bar-video").classList.remove("indeterminate");
      $("bar-video").style.width = "100%";
      showOriginalPreview(videoFile);
      const ve = $("vid-enhanced");
      ve.src = j.video + "?t=" + Date.now();
      $("dl-video-link").href = j.video;
      $("results-video").hidden = false; $("dl-video").hidden = false;
      let msg = `Done – ${j.frames} frames at ${j.fps} fps (${j.mean_ms} ms/frame pipeline time).`;
      if (!j.playable_in_browser) msg += " This MP4 may not play in the browser – download it and open in a video player.";
      setStatus("video", msg);
    } catch (e) {
      $("progress-video").hidden = true;
      setStatus("video", e.message, true);
    } finally {
      $("go-video").disabled = false;
    }
  });
})();
