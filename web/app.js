// LabVision Copilot — vanilla JS, no build step
const $ = s => document.querySelector(s);
const fileInput = $("#fileInput"), dropzone = $("#dropzone"), previewThumb = $("#previewThumb");
const runBtn = $("#runBtn"), requestInput = $("#requestInput"), leftStatus = $("#leftStatus");
const micBtn = $("#micBtn"), micStatus = $("#micStatus");
const exampleThumbs = $("#exampleThumbs");
const emptyState = $("#emptyState"), loadingState = $("#loadingState"), loadingText = $("#loadingText"), errorState = $("#errorState"), errorMsg = $("#errorMsg"), errorHint = $("#errorHint"), resultState = $("#resultState");
const countBig = $("#countBig"), countSub = $("#countSub"), statusPill = $("#statusPill");
const viewer = $("#viewer"), canvasWrap = $("#canvasWrap"), clickSpinner = $("#clickSpinner");
const masksToggle = $("#masksToggle"), boxesToggle = $("#boxesToggle");
const modeAdd = $("#modeAdd"), modeRemove = $("#modeRemove");
const traceList = $("#traceList"), measureTable = $("#measureTable"), csvLink = $("#csvLink"), colonyTypesEl = $("#colonyTypes"), reportText = $("#reportText");
const cameraBtn = $("#cameraBtn"), cameraWrap = $("#cameraWrap"), cameraVideo = $("#cameraVideo"), cameraCanvas = $("#cameraCanvas"), cameraCaptureBtn = $("#cameraCaptureBtn"), cameraCancelBtn = $("#cameraCancelBtn");

let selectedFile = null; // Blob
let lastResult = null;   // /api/analyze JSON
let correctMode = "add";
let imageEl = null, masksEl = null; // HTMLImage for canvas draw
let ctx = viewer.getContext("2d");

// ---------- helpers
function showState(which){
  emptyState.hidden = which!=="empty";
  loadingState.hidden = which!=="loading";
  errorState.hidden = which!=="error";
  resultState.hidden = which!=="result";
}
function setLeftStatus(t){ leftStatus.textContent = t||""; }
function setError(err, hint){ errorMsg.textContent = err||"Error"; errorHint.textContent = hint||""; showState("error"); }
function pillClass(status){
  const s=(status||"ok").toLowerCase();
  if(s==="tntc") return "tntc";
  if(s==="blurry"||s==="glare"||s==="empty") return s;
  if(s==="ok") return "ok";
  return "ok";
}
function isTNTC(status, count){ return status==="tntc" || count>300; }

// ---------- file handling
function setFile(blob, name){
  selectedFile = blob;
  selectedFile._name = name || "image.jpg";
  // preview
  const url = URL.createObjectURL(blob);
  previewThumb.src = url;
  previewThumb.hidden = false;
  dropzone.querySelector(".dz-text").textContent = name || "Image selected";
  // hide camera if open
  stopCamera();
}

dropzone.addEventListener("click", ()=> fileInput.click());
dropzone.addEventListener("keydown", e=>{ if(e.key==="Enter"||e.key===" "){ e.preventDefault(); fileInput.click(); }});
dropzone.addEventListener("dragover", e=>{ e.preventDefault(); dropzone.classList.add("dragover"); });
dropzone.addEventListener("dragleave", ()=> dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", e=>{
  e.preventDefault(); dropzone.classList.remove("dragover");
  const f=e.dataTransfer.files[0]; if(f) setFile(f, f.name);
});
fileInput.addEventListener("change", ()=>{
  const f=fileInput.files[0]; if(f) setFile(f, f.name);
});
// paste
document.addEventListener("paste", e=>{
  const f=[...(e.clipboardData?.items||[])].find(i=>i.kind==="file");
  if(f){ const file=f.getAsFile(); if(file) setFile(file, file.name); }
});

// ---------- camera
let camStream=null;
async function startCamera(){
  try{
    camStream = await navigator.mediaDevices.getUserMedia({video:{facingMode:"environment"}});
    cameraVideo.srcObject = camStream;
    cameraWrap.hidden=false; cameraCaptureBtn.hidden=false; cameraCancelBtn.hidden=false;
    cameraBtn.textContent="Camera on";
  }catch(e){ setLeftStatus("Camera unavailable: "+e.message); }
}
function stopCamera(){
  if(camStream){ camStream.getTracks().forEach(t=>t.stop()); camStream=null; }
  cameraWrap.hidden=true; cameraCaptureBtn.hidden=true; cameraCancelBtn.hidden=true;
  cameraBtn.textContent="Use camera";
}
cameraBtn.addEventListener("click", ()=> camStream?stopCamera():startCamera());
cameraCancelBtn.addEventListener("click", stopCamera);
cameraCaptureBtn.addEventListener("click", ()=>{
  const w=cameraVideo.videoWidth, h=cameraVideo.videoHeight;
  if(!w) return;
  cameraCanvas.width=w; cameraCanvas.height=h;
  cameraCanvas.getContext("2d").drawImage(cameraVideo,0,0,w,h);
  cameraCanvas.toBlob(blob=>{ if(blob) setFile(blob, "camera.jpg"); }, "image/jpeg", 0.9);
  stopCamera();
});

// ---------- examples
async function loadExamples(){
  try{
    const r=await fetch("/api/examples"); const list=await r.json();
    if(!Array.isArray(list)||!list.length){
      exampleThumbs.innerHTML='<span class="muted small">No local examples found</span>';
      return;
    }
    exampleThumbs.innerHTML="";
    for(const ex of list){
      const wrap=document.createElement("div");
      const img=document.createElement("img");
      img.className="thumb"; img.alt=ex.name||ex.path; img.loading="lazy";
      // use api example image? spec says list of local images if present in data/ — server returns path
      // but browser can't load data/ directly; we fetch via blob from path if we had endpoint
      // fallback: show placeholder and fetch on click via /api/examples image proxy
      // we try to load from same path if server serves it; otherwise fetch blob via GET
      // For now assign src to try direct; server doesn't expose data/ — we'll handle click by fetching image
      // Use a data thumbnail: try to fetch thumbnail via fetch image blob URL
      // Create object url by fetching image through a temp run? Instead just show label and click to load
      img.src = ""; // placeholder
      // try to fetch the image file via a simple fetch to a path that may be served? we don't have that route.
      // So we show initials and rely on click to load
      img.style.display="grid"; img.style.placeItems="center";
      img.title = ex.name || ex.path;
      // attempt to load via /api/examples image: we need an endpoint that serves example image
      // workaround: fetch the file via a new endpoint? But we don't have it — we can fetch via GET to ex.path if server serves static?
      // For now, create a canvas placeholder with name
      const ph=document.createElement("div");
      ph.className="thumb"; ph.style.display="grid"; ph.style.placeItems="center"; ph.style.fontSize=".7rem"; ph.style.cursor="pointer"; ph.textContent=(ex.name||"img").slice(0,8);
      ph.title=ex.path;
      ph.addEventListener("click", async ()=>{
        document.querySelectorAll(".thumb.is-selected").forEach(e=>e.classList.remove("is-selected"));
        ph.classList.add("is-selected");
        // fetch image blob from server: we try /api/examples/image?name=... or direct path
        // Since server doesn't expose data/, we try to fetch via a dedicated endpoint — add fallback: try fetch(ex.path)
        try{
          let blob=null;
          // try direct path (if server serves project root)
          try{ const rr=await fetch(ex.path); if(rr.ok) blob=await rr.blob(); }catch(_){}
          if(!blob){
            // try via api: GET /api/examples/file?path=...
            try{ const rr2=await fetch("/api/examples/file?path="+encodeURIComponent(ex.path)); if(rr2.ok) blob=await rr2.blob(); }catch(_){}
          }
          if(blob){ setFile(blob, ex.name||"example.jpg"); setLeftStatus("Example loaded — click Run"); }
          else { setLeftStatus("Example not available locally — using placeholder"); }
        }catch(e){ setLeftStatus("Failed to load example"); }
      });
      wrap.appendChild(ph);
      const lab=document.createElement("div"); lab.className="thumb-label"; lab.textContent=ex.name||"example";
      wrap.appendChild(lab);
      exampleThumbs.appendChild(wrap);

      // try to load thumbnail image if fetchable
      (async()=>{
        try{
          const rr=await fetch(ex.path);
          if(rr.ok){ const blob=await rr.blob(); ph.textContent=""; const realImg=document.createElement("img"); realImg.className="thumb"; realImg.src=URL.createObjectURL(blob); realImg.alt=ex.name; realImg.style.cursor="pointer";
            realImg.addEventListener("click", async()=>{
              document.querySelectorAll(".thumb.is-selected").forEach(e=>e.classList.remove("is-selected"));
              realImg.classList.add("is-selected");
              setFile(blob, ex.name||"example.jpg"); setLeftStatus("Example loaded — click Run");
            });
            ph.replaceWith(realImg);
          }
        }catch(_){}
      })();
    }
    // preselect first for demo (Step 4: example 14684 preselected)
    // Auto-load first example blob so Run works with one click
    if(list.length){
      try{
        const ex=list[0];
        try{
          const rr=await fetch(ex.path);
          if(rr.ok){ const blob=await rr.blob(); setFile(blob, ex.name||"example.jpg"); setLeftStatus("Example "+(ex.name||"")+ " ready — click Run"); document.querySelector(".thumb")?.classList.add("is-selected"); }
        }catch(_){}
      }catch(_){}
    }
  }catch(_){ exampleThumbs.innerHTML='<span class="muted small">Examples unavailable</span>'; }
}
loadExamples();

// ---------- run analyze
async function doAnalyze(){
  if(!selectedFile){ setError("No image", "Drop an image or pick an example."); return; }
  const req = requestInput.value.trim() || "count colonies";
  showState("loading"); loadingText.textContent="Analysing… check_quality → segment → measure";
  runBtn.disabled=true; setLeftStatus("");
  try{
    const fd=new FormData();
    fd.append("image", selectedFile, selectedFile._name||"image.jpg");
    fd.append("request", req);
    const r=await fetch("/api/analyze", {method:"POST", body:fd});
    if(!r.ok){
      let j={}; try{ j=await r.json(); }catch(_){ j={error:await r.text()}; }
      const detail=j.detail||j;
      throw new Error(detail.error||detail.detail||("HTTP "+r.status));
    }
    const data=await r.json();
    lastResult=data;
    renderResult(data);
    showState("result");
    setLeftStatus("");
  }catch(e){
    const msg=e.message||String(e);
    let hint="Check API keys in .env and try again.";
    if(msg.toLowerCase().includes("meta_api_key")) hint="Missing META_API_KEY in .env";
    setError(msg.slice(0,400), hint);
  }finally{ runBtn.disabled=false; }
}
runBtn.addEventListener("click", doAnalyze);

// ---------- render result
function renderResult(data){
  // hero
  const isT=isTNTC(data.status, data.count);
  countBig.textContent = data.count_label || (isT ? "TNTC >300" : data.count+" CFU");
  countSub.textContent = (data.image_w&&data.image_h)? `${data.image_w}×${data.image_h} · ${data.duration_ms||""} ms` : "";
  statusPill.textContent = data.status==="tntc" ? "TNTC — dilute & re-plate" : data.status==="blurry" ? "blurry — re-photo" : data.status==="glare" ? "glare" : data.status==="empty" ? "empty" : "OK";
  statusPill.className="pill "+pillClass(data.status);

  // viewer
  loadViewerImages(data);

  // trace
  renderTrace(data.trace||[]);

  // measurements
  renderMeasurements(data.measurements||[], data.csv_url);

  // colony types
  renderColonyTypes(data.colony_types||[]);

  // report
  reportText.textContent = data.report || "";
}

function renderTrace(trace){
  traceList.innerHTML="";
  if(!trace.length){ traceList.innerHTML='<li class="muted">No trace</li>'; return; }
  for(let i=0;i<trace.length;i++){
    const t=trace[i];
    const li=document.createElement("li");
    const icon=document.createElement("div"); icon.className="tl-icon "+(t.status||"ok");
    icon.textContent=t.status==="ok"?"✓":t.status==="warn"?"!":"✕";
    const body=document.createElement("div"); body.className="tl-body";
    const name=document.createElement("div"); name.className="tl-name"; name.textContent=(i+1)+". "+(t.name||"step");
    const kv=document.createElement("div"); kv.className="tl-kv";
    // pick key values
    const parts=[];
    if(t.blur_score!==undefined) parts.push("blur "+t.blur_score);
    if(t.glare_fraction!==undefined) parts.push("glare "+t.glare_fraction);
    if(t.prompt_used) parts.push("prompt "+t.prompt_used);
    if(t.retry!==undefined) parts.push("retry "+(t.retry?"yes":"no"));
    if(t.plate_diameter_px) parts.push("plate "+t.plate_diameter_px+"px");
    if(t.mm_per_px) parts.push(t.mm_per_px+" mm/px");
    if(t.boxes!==undefined) parts.push(t.boxes+" boxes");
    if(t.count!==undefined) parts.push("count "+t.count);
    if(t.duration_ms!==undefined) parts.push(t.duration_ms+"ms");
    if(t.model) parts.push(t.model);
    if(t.error) parts.push("error: "+t.error);
    kv.textContent=parts.join(" · ")||JSON.stringify(t);
    body.appendChild(name); body.appendChild(kv);
    li.appendChild(icon); li.appendChild(body);
    // stagger animation if not reduced motion
    if(!window.matchMedia("(prefers-reduced-motion: reduce)").matches){
      li.style.opacity="0"; li.style.transform="translateY(6px)"; li.style.transition="opacity .25s, transform .25s";
      setTimeout(()=>{ li.style.opacity="1"; li.style.transform="none"; }, i*80);
    }
    traceList.appendChild(li);
  }
}

function renderMeasurements(rows, csvUrl){
  const tbody=measureTable.querySelector("tbody"); tbody.innerHTML="";
  if(!rows.length){ tbody.innerHTML='<tr><td colspan="5" class="muted">No measurements</td></tr>'; csvLink.hidden=true; return; }
  for(const r of rows){
    const tr=document.createElement("tr");
    tr.innerHTML=`<td>${r.id??""}</td><td>${r.area_px??r.area??""}</td><td>${r.diameter_px??r.diameter??""}</td><td>${r.area_mm2??""}</td><td>${r.diameter_mm??""}</td>`;
    tbody.appendChild(tr);
  }
  if(csvUrl){ csvLink.href=csvUrl; csvLink.hidden=false; } else csvLink.hidden=true;
}

function renderColonyTypes(types){
  if(!types.length){ colonyTypesEl.innerHTML='<p class="muted">No colony types — add "characterize" to your request (e.g. "count and characterize colonies").</p>'; return; }
  let html='<div class="table-wrap"><table><thead><tr><th>Type</th><th>Description</th><th>Count</th><th>Mean Ø (px)</th></tr></thead><tbody>';
  for(const t of types){
    html+=`<tr><td>${t.label||t.type||""}</td><td>${t.description||""}</td><td>${t.count??""}</td><td>${t.mean_diameter_px??t.mean_diameter??""}</td></tr>`;
  }
  html+='</tbody></table></div>';
  colonyTypesEl.innerHTML=html;
}

// ---------- canvas viewer
function loadViewerImages(data){
  // create image objects for drawing
  const imgUrl = data.image_url;
  const masksUrl = data.masks_url;
  imageEl = new Image(); imageEl.crossOrigin="anonymous";
  masksEl = null;
  let imgLoaded=false, masksLoaded=false;
  function tryDraw(){ if(imgLoaded) drawViewer(); }

  imageEl.onload=()=>{ imgLoaded=true; tryDraw(); };
  imageEl.onerror=()=>{ imgLoaded=true; tryDraw(); };
  imageEl.src = imgUrl;

  if(masksUrl && masksToggle.checked){
    masksEl = new Image(); masksEl.crossOrigin="anonymous";
    masksEl.onload=()=>{ masksLoaded=true; tryDraw(); };
    masksEl.onerror=()=>{ masksLoaded=false; tryDraw(); };
    masksEl.src = masksUrl;
  } else {
    tryDraw();
  }
  // store boxes for drawing
  if(data.boxes) lastResult.boxes = data.boxes;
  if(data.image_w) lastResult.image_w = data.image_w;
  if(data.image_h) lastResult.image_h = data.image_h;
}

function drawViewer(){
  if(!lastResult || !imageEl) return;
  const w = lastResult.image_w || imageEl.naturalWidth;
  const h = lastResult.image_h || imageEl.naturalHeight;
  // size canvas to image size (let CSS scale down)
  viewer.width = w; viewer.height = h;
  viewer.style.width="100%"; viewer.style.height="auto";
  ctx.clearRect(0,0,w,h);
  try{ ctx.drawImage(imageEl,0,0,w,h); }catch(_){}
  if(masksToggle.checked && masksEl && masksEl.complete && masksEl.naturalWidth){
    ctx.globalAlpha=0.55;
    try{ ctx.drawImage(masksEl,0,0,w,h); }catch(_){}
    ctx.globalAlpha=1;
  }
  if(boxesToggle.checked && lastResult.boxes){
    ctx.strokeStyle="#1F6F6A"; ctx.lineWidth=Math.max(1, Math.round(w/700));
    for(const b of lastResult.boxes){
      ctx.strokeRect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
    }
  }
  // markers for last click (optional)
  if(lastResult._lastClick){
    const {x,y}=lastResult._lastClick;
    ctx.fillStyle="#C8892B"; ctx.beginPath(); ctx.arc(x,y, Math.max(4, w/250),0,Math.PI*2); ctx.fill();
    ctx.strokeStyle="#fff"; ctx.lineWidth=1.5; ctx.stroke();
  }
}
masksToggle.addEventListener("change", drawViewer);
boxesToggle.addEventListener("change", drawViewer);

// resize observer to keep canvas crisp — CSS scales, canvas buffer stays native
window.addEventListener("resize", ()=>{ if(lastResult) drawViewer(); });

// ---------- click-to-correct
function getClickCoords(e){
  const rect = viewer.getBoundingClientRect();
  const w = lastResult.image_w, h = lastResult.image_h;
  const x = (e.clientX - rect.left) * w / rect.width;
  const y = (e.clientY - rect.top) * h / rect.height;
  return {x: Math.floor(Math.max(0, Math.min(w-1, x))), y: Math.floor(Math.max(0, Math.min(h-1, y)))};
}
viewer.addEventListener("click", async (e)=>{
  if(!lastResult || !lastResult.run_id) return;
  const {x,y}=getClickCoords(e);
  lastResult._lastClick={x,y}; drawViewer();
  clickSpinner.hidden=false;
  try{
    const r=await fetch("/api/correct", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({run_id:lastResult.run_id, x,y, mode:correctMode})});
    if(!r.ok){
      let j={}; try{ j=await r.json(); }catch(_){}
      throw new Error((j.detail&&j.detail.error)||j.error||("HTTP "+r.status));
    }
    const data=await r.json();
    // merge boxes / count — server returns updated counts and boxes
    lastResult.count = data.count; lastResult.count_label = data.count_label; lastResult.boxes = data.boxes;
    if(data.status) lastResult.status = data.status;
    if(data.measurements) lastResult.measurements = data.measurements;
    if(data.correction) lastResult._lastCorrection = data.correction;
    // also update hero + table
    countBig.textContent = data.count_label || data.count+" CFU";
    statusPill.textContent = (data.status==="tntc"?"TNTC — dilute & re-plate":data.status||"ok");
    statusPill.className="pill "+pillClass(data.status);
    renderMeasurements(data.measurements||[], data.csv_url||lastResult.csv_url);
    // need to reload image? masks.png updated on server — force reload
    if(lastResult.masks_url) masksEl=null; // will reload via timestamp bust
    // reload masks image
    const bust = "?t="+Date.now();
    if(masksToggle.checked){
      masksEl = new Image(); masksEl.crossOrigin="anonymous";
      masksEl.onload=drawViewer; masksEl.src = lastResult.masks_url + bust;
    }
    // update boxes immediately (without waiting for masks)
    drawViewer();
  }catch(err){
    setLeftStatus(err.message.slice(0,120));
  }finally{ clickSpinner.hidden=true; }
});

modeAdd.addEventListener("click", ()=>{ correctMode="add"; modeAdd.classList.add("is-active"); modeRemove.classList.remove("is-active"); });
modeRemove.addEventListener("click", ()=>{ correctMode="remove"; modeRemove.classList.add("is-active"); modeAdd.classList.remove("is-active"); });

// ---------- tabs
document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab").forEach(b=>{ b.classList.remove("is-active"); b.setAttribute("aria-selected","false"); });
    btn.classList.add("is-active"); btn.setAttribute("aria-selected","true");
    const t=btn.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach(p=> p.hidden=true);
    $("#tab-"+t).hidden=false;
  });
});

// ---------- mic — Web Audio API, resample to 24k mono, WAV encode, POST /api/transcribe
let recState=null;
function encodeWAV(pcmFloat32, sampleRate){
  // pcmFloat32: Float32Array mono 24k, values -1..1 -> int16
  const numSamples=pcmFloat32.length;
  const buf=new ArrayBuffer(44+numSamples*2);
  const view=new DataView(buf);
  const writeStr=(off,s)=>{ for(let i=0;i<s.length;i++) view.setUint8(off+i,s.charCodeAt(i)); };
  writeStr(0,"RIFF"); view.setUint32(4,36+numSamples*2,true); writeStr(8,"WAVE"); writeStr(12,"fmt ");
  view.setUint32(16,16,true); view.setUint16(20,1,true); view.setUint16(22,1,true); view.setUint32(24,sampleRate,true);
  view.setUint32(28,sampleRate*2,true); view.setUint16(32,2,true); view.setUint16(34,16,true); writeStr(36,"data"); view.setUint32(40,numSamples*2,true);
  let off=44;
  for(let i=0;i<numSamples;i++){ let s=Math.max(-1,Math.min(1,pcmFloat32[i])); view.setInt16(off, s<0?s*0x8000:s*0x7FFF,true); off+=2; }
  return new Blob([view],{type:"audio/wav"});
}
function resampleLinear(input, inRate, outRate){
  if(inRate===outRate) return input;
  const ratio=inRate/outRate;
  const outLen=Math.round(input.length/ratio);
  const out=new Float32Array(outLen);
  for(let i=0;i<outLen;i++){ const idx=i*ratio; const lo=Math.floor(idx), hi=Math.min(lo+1,input.length-1); const frac=idx-lo; out[i]=input[lo]*(1-frac)+input[hi]*frac; }
  return out;
}
async function startRec(){
  try{
    const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true}});
    const AudioCtx=window.AudioContext||window.webkitAudioContext;
    const ac=new AudioCtx();
    const src=ac.createMediaStreamSource(stream);
    const proc=ac.createScriptProcessor(4096,1,1);
    let chunks=[];
    proc.onaudioprocess=e=>{
      const ch=e.inputBuffer.getChannelData(0);
      chunks.push(new Float32Array(ch));
    };
    src.connect(proc); proc.connect(ac.destination);
    recState={ac, stream, proc, src, chunks, sampleRate:ac.sampleRate};
    micBtn.classList.add("is-recording"); micStatus.textContent="Recording… click again to stop";
  }catch(e){ micStatus.textContent="Mic error: "+e.message; }
}
async function stopRec(){
  if(!recState) return;
  const {ac, stream, proc, src, chunks, sampleRate}=recState;
  try{ src.disconnect(); proc.disconnect(); }catch(_){}
  stream.getTracks().forEach(t=>t.stop());
  micBtn.classList.remove("is-recording");
  micStatus.textContent="Transcribing…";
  // concat
  let total=0; for(const c of chunks) total+=c.length;
  const all=new Float32Array(total); let o=0; for(const c of chunks){ all.set(c,o); o+=c.length; }
  const resampled=resampleLinear(all, sampleRate, 24000);
  const wavBlob=encodeWAV(resampled, 24000);
  recState=null;
  try{
    const fd=new FormData(); fd.append("audio", wavBlob, "audio.wav");
    const r=await fetch("/api/transcribe", {method:"POST", body:fd});
    if(!r.ok){ let j={}; try{ j=await r.json(); }catch(_){} throw new Error((j.detail&&j.detail.error)||j.error||"transcribe failed"); }
    const j=await r.json();
    if(j.transcript){ requestInput.value=j.transcript; micStatus.textContent="Done — edit if needed, then Run"; }
    else micStatus.textContent="No transcript";
  }catch(e){ micStatus.textContent="Transcribe error: "+e.message.slice(0,120); }
}
micBtn.addEventListener("click", ()=>{
  if(recState) stopRec(); else startRec();
});

// keyboard: Enter in request runs
requestInput.addEventListener("keydown", e=>{ if(e.key==="Enter" && (e.metaKey||e.ctrlKey)){ doAnalyze(); }});
