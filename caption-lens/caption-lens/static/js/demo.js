const API_URL = "/api/caption";
let file = null, url = null;
/* upload */
const drop = $("#drop"), input = $("#file");
drop.addEventListener("click", () => input.click());
drop.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); input.click(); } });
["dragenter","dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => setFile(e.dataTransfer.files[0]));
input.addEventListener("change", () => setFile(input.files[0]));
$("#beams").addEventListener("input", e => $("#beamOut").textContent = e.target.value);

function setFile(f){
  if (!f) return;
  if (!/^image\/(jpeg|png|webp)$/.test(f.type)) return say("That file type isn't supported. Choose a JPG, PNG, or WebP image.");
  if (f.size > 10 * 1024 * 1024) return say("That image is larger than 10 MB. Choose a smaller one.");
  file = f; if (url) URL.revokeObjectURL(url); url = URL.createObjectURL(f);
  $("#preview").src = url; $("#prompt").classList.add("d-none"); $("#frame").classList.remove("d-none");
  $("#go").disabled = false; $("#reset").disabled = false; clearCaps(); say("Image ready. Press Generate captions.");
}
function say(m){ $("#status").textContent = m; }
function clearCaps(){ ["#capA","#capB"].forEach(s => { const el = $(s); el._id = (el._id||0)+1; el.className = "cap idle"; el.textContent = "The caption will appear here."; }); $("#banner").classList.add("d-none"); }
$("#reset").addEventListener("click", () => {
  file = null; input.value = ""; $("#preview").removeAttribute("src");
  $("#frame").classList.add("d-none"); $("#prompt").classList.remove("d-none");
  $("#go").disabled = true; $("#reset").disabled = true; clearCaps(); say("Choose an image to begin.");
});

/* generate */
const samples = [["a dog running on a grassy field","a dog is running in the grass"],["a man riding a bike down a street","a man is riding a bicycle"],["a group of people standing on a beach","people are standing near the water"]];
$("#go").addEventListener("click", async () => {
  const go = $("#go"), frame = $("#frame"), stages = ["Preprocessing the image…","Extracting visual features…","Writing the captions word by word…"];
  go.disabled = true; $("#reset").disabled = true; frame.classList.add("busy"); clearCaps();
  ["#capA","#capB"].forEach(s => { $(s).className = "cap idle"; $(s).innerHTML = '<div class="sk"></div><div class="sk w-75"></div>'; });
  let k = 0; say(stages[0]); const tick = setInterval(() => say(stages[++k % stages.length]), 900);
  let out, demo = false;
  try{
    const fd = new FormData(); fd.append("image", file); fd.append("beams", $("#beams").value);
    const ctl = new AbortController(), to = setTimeout(() => ctl.abort(), 90000);
    const r = await fetch(API_URL, { method:"POST", body:fd, signal:ctl.signal }); clearTimeout(to);
    if (!r.ok) throw new Error("http");
    out = await r.json();
    if (!out.blip && !out.gru) throw new Error("shape");
  }catch(err){
    if (err.message === "http" || err.message === "shape"){
      clearInterval(tick); frame.classList.remove("busy"); clearCaps(); go.disabled = false; $("#reset").disabled = false;
      return say("The server replied with an error. Check the backend console for details.");
    }
    demo = true; await new Promise(r => setTimeout(r, 1800));
    const s = samples[Math.floor(Math.random() * samples.length)]; out = { blip:s[0], gru:s[1] };
  }
  clearInterval(tick); frame.classList.remove("busy");
  const b = $("#banner");
  if (demo){ b.textContent = "Backend not connected, so these are sample captions, not real model output. Run python app.py and open the site from http://127.0.0.1:5000 to see real results."; b.classList.remove("d-none"); }
  say(demo ? "Showing sample captions." : "Done. Compare the two captions.");
  await Promise.all([type($("#capA"), out.blip || "BLIP is unavailable. Check the server console."), type($("#capB"), out.gru || "The GRU model is not trained yet. Save checkpoints/gru_best.pt to enable it.")]);
  go.disabled = false; $("#reset").disabled = false;
});

/* copy buttons */
document.querySelectorAll(".copy").forEach(btn => btn.addEventListener("click", async () => {
  const el = $("#" + btn.dataset.t); if (el.classList.contains("idle")) return;
  try{ await navigator.clipboard.writeText(el.textContent); btn.innerHTML = '<i class="bi bi-check2"></i>'; setTimeout(() => btn.innerHTML = '<i class="bi bi-clipboard"></i>', 1200); }catch(e){}
}));
