const $ = s => document.querySelector(s);
const fileInput = $("#fileInput"), dropzone = $("#dropzone"), previewThumb = $("#previewThumb");
const runBtn = $("#runBtn"), requestInput = $("#requestInput"), leftStatus = $("#leftStatus");
const micBtn = $("#micBtn"), micStatus = $("#micStatus");
const exampleThumbs = $("#exampleThumbs");
const emptyState = $("#emptyState"), loadingState = $("#loadingState"), loadingText = $("#loadingText"), errorState = $("#errorState"), errorMsg = $("#errorMsg"), errorHint = $("#errorHint"), resultState = $("#resultState");
const liveProgress = $("#liveProgress"), liveSteps = $("#liveSteps");
const countBig = $("#countBig"), countSub = $("#countSub"), statusPill = $("#statusPill");
const viewer = $("#viewer"), canvasWrap = $("#canvasWrap"), clickSpinner = $("#clickSpinner");
const masksToggle = $("#masksToggle"), boxesToggle = $("#boxesToggle");
const modeAdd = $("#modeAdd"), modeRemove = $("#modeRemove"), modeManual = $("#modeManual");
const traceList = $("#traceList"), measureTable = $("#measureTable"), csvLink = $("#csvLink"), colonyTypesEl = $("#colonyTypes"), reportText = $("#reportText");
const cameraBtn = $("#cameraBtn"), cameraWrap = $("#cameraWrap"), cameraVideo = $("#cameraVideo"), cameraCanvas = $("#cameraCanvas"), cameraCaptureBtn = $("#cameraCaptureBtn"), cameraCancelBtn = $("#cameraCancelBtn");

let selectedFile = null;
let lastResult = null;
let correctMode = "add";
let imageEl = null, masksEl = null;
let ctx = viewer.getContext("2d");

function showState(which){
  emptyState.hidden = which!=="empty";
  loadingState.hidden = which!=="loading";
  // How it was verified always visible per mock
  if(liveProgress) liveProgress.hidden = false;
  errorState.hidden = which!=="error";
  resultState.hidden = which!=="result";
}
function setLeftStatus(t){ leftStatus.textContent = t||""; }
function setError(err, hint){ errorMsg.textContent = err||"Error"; errorHint.textContent = hint||""; showState("error"); }
function pillClass(status){
  const s=(status||"ok").toLowerCase();
  if(s==="tntc") return "tntc";
  if(s==="re-photo") return "rephoto";
  if(s==="empty") return "empty";
  if(s==="ok") return "ok";
  // legacy: blurry/glare map to re-photo
  if(s==="blurry"||s==="glare") return "rephoto";
  return "ok";
}
function isTNTC(status, count){ return (status||"").toLowerCase()==="tntc" || count>300; }

function setFile(blob, name){
  selectedFile = blob;
  selectedFile._name = name || "image.jpg";
  const url = URL.createObjectURL(blob);
  previewThumb.src = url;
  previewThumb.hidden = false;
  dropzone.querySelector(".dz-text").textContent = name || "Image selected";
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
document.addEventListener("paste", e=>{
  const f=[...(e.clipboardData?.items||[])].find(i=>i.kind==="file");
  if(f){ const file=f.getAsFile(); if(file) setFile(file, file.name); }
});

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

// suggested questions chips
const _suggested = [
  "count colonies",
  "count colonies larger than 2 mm",
  "count colonies smaller than 1 mm",
  "is this plate countable?",
  "measure colony sizes",
  "count black colonies (experimental)",
  "count yellow colonies (experimental)",
];
(function initChips(){
  const c = document.getElementById("suggestedChips");
  if(!c) return;
  for(const txt of _suggested){
    const b=document.createElement("button");
    b.type="button"; b.className="sug-chip"+(txt.includes("experimental")?" experimental":"");
    b.textContent=txt; b.title=txt.includes("experimental")?"Experimental — not benchmarked":"Click to fill request";
    b.addEventListener("click", ()=>{ requestInput.value=txt.replace(" (experimental)",""); requestInput.focus(); });
    c.appendChild(b);
  }
})();

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
      img.className="thumb";
      img.alt=ex.name||ex.path;
      img.loading="lazy";
      img.src="/api/examples/file?path="+encodeURIComponent(ex.path);
      img.title=ex.name||ex.path;
      img.addEventListener("click", async ()=>{
        document.querySelectorAll(".thumb.is-selected").forEach(e=>e.classList.remove("is-selected"));
        img.classList.add("is-selected");
        try{
          const rr=await fetch("/api/examples/file?path="+encodeURIComponent(ex.path));
          if(!rr.ok) throw new Error("fetch failed");
          const blob=await rr.blob();
          setFile(blob, ex.name||"example.jpg");
          setLeftStatus("Example loaded — click Run");
        }catch(_){ setLeftStatus("Failed to load example"); }
      });
      // fallback if image fails to load
      img.addEventListener("error", ()=>{ img.style.display="none"; });
      wrap.appendChild(img);
      const lab=document.createElement("div"); lab.className="thumb-label"; lab.textContent=ex.name||"example";
      wrap.appendChild(lab);
      exampleThumbs.appendChild(wrap);
    }
    // ?example=14684 preload (from landing)
    const qp=new URLSearchParams(location.search).get("example");
    let preselect=list[0];
    if(qp){
      const found=list.find(e=> e.name.includes(qp) || e.path.includes(qp));
      if(found) preselect=found;
    }
    if(preselect){
      try{
        const rr=await fetch("/api/examples/file?path="+encodeURIComponent(preselect.path));
        if(rr.ok){
          const blob=await rr.blob();
          setFile(blob, preselect.name||"example.jpg");
          setLeftStatus("Example "+(preselect.name||"")+ " ready — click Run");
          // mark selected thumb after render (next tick)
          setTimeout(()=>{
            const thumbs=document.querySelectorAll(".thumb");
            for(const t of thumbs){ if(t.alt===preselect.name || t.title===preselect.name) t.classList.add("is-selected"); }
            if(!document.querySelector(".thumb.is-selected")) document.querySelector(".thumb")?.classList.add("is-selected");
          }, 100);
        }
      }catch(_){}
    }
  }catch(_){ exampleThumbs.innerHTML='<span class="muted small">Examples unavailable</span>'; }
}
loadExamples();

// queue detail: ?run_id=xxx
(async function tryLoadRunId(){
  const rid=new URLSearchParams(location.search).get("run_id");
  if(!rid) return;
  try{
    const r=await fetch(`/api/runs/${encodeURIComponent(rid)}`);
    if(!r.ok) throw new Error("run not found");
    const data=await r.json();
    lastResult=data;
    renderResult(data);
    showState("result");
    setLeftStatus(`Loaded ${data.plate_id||rid} from queue — you can correct by click/box`);
  }catch(e){ /* keep empty state */ }
})();

const _liveMap = new Map();
function clearLive(){
  _liveMap.clear();
  if(liveSteps) liveSteps.innerHTML="";
}
function upsertLiveStep(evt){
  if(!liveSteps) return;
  let li = _liveMap.get(evt.step);
  if(!li){
    li=document.createElement("li");
    const icon=document.createElement("div"); icon.className="tl-icon running"; icon.textContent="…";
    const body=document.createElement("div"); body.className="tl-body";
    const name=document.createElement("div"); name.className="tl-name";
    const kv=document.createElement("div"); kv.className="tl-kv";
    const badge=document.createElement("span"); badge.className="tl-badge "+(evt.decided_by||"rule"); badge.textContent=evt.decided_by||"";
    body.append(name, kv);
    li.append(icon, body, badge);
    liveSteps.appendChild(li);
    _liveMap.set(evt.step, {li, icon, name, kv, badge});
  }
  const e=_liveMap.get(evt.step);
  e.icon.className="tl-icon "+(evt.status==="running"?"running":evt.status||"ok");
  e.icon.textContent=evt.status==="running"?"…":evt.status==="ok"?"✓":evt.status==="warn"?"!":"✕";
  e.name.textContent=evt.message||evt.step;
  const vals=evt.values||{};
  const parts=[];
  if(vals.blur_score!==undefined) parts.push("blur "+vals.blur_score);
  if(vals.glare_fraction!==undefined) parts.push("glare "+vals.glare_fraction);
  if(vals.prompt) parts.push("prompt "+vals.prompt);
  if(vals.prompt_used) parts.push("prompt "+vals.prompt_used);
  if(vals.boxes!==undefined) parts.push(vals.boxes+" boxes");
  if(vals.mean_confidence!=null) parts.push(`conf ${Number(vals.mean_confidence).toFixed(2)}${vals.low_confidence?" low":""}`);
  if(vals.count!==undefined) parts.push("count "+vals.count);
  if(vals.plate_diameter_px) parts.push("plate "+vals.plate_diameter_px+"px");
  if(vals.mm_per_px) parts.push(vals.mm_per_px+" mm/px");
  if(vals.verdict) parts.push(vals.verdict);
  e.kv.textContent=parts.join(" · ");
  e.badge.className="tl-badge "+(evt.decided_by||"rule");
  e.badge.textContent=evt.decided_by||"";
  e.badge.hidden=!evt.decided_by;
}

async function doAnalyzeStream(fd){
  const r=await fetch("/api/analyze_stream", {method:"POST", body:fd});
  if(!r.ok || !r.body || !r.headers.get("content-type")?.includes("text/event-stream")){
    throw new Error("stream not available");
  }
  const reader=r.body.getReader();
  const decoder=new TextDecoder();
  let buf="";
  let finalData=null;
  while(true){
    const {value, done}=await reader.read();
    if(done) break;
    buf+=decoder.decode(value, {stream:true});
    let idx;
    while((idx=buf.indexOf("\n\n"))!==-1){
      const chunk=buf.slice(0, idx);
      buf=buf.slice(idx+2);
      for(const line of chunk.split("\n")){
        if(!line.startsWith("data:")) continue;
        const obj=JSON.parse(line.slice(5).trim());
        if(obj.type==="step"){
          upsertLiveStep(obj);
        } else if(obj.type==="result"){
          finalData=obj.data;
        } else if(obj.type==="error"){
          throw new Error(obj.error||"stream error");
        }
      }
    }
  }
  if(!finalData) throw new Error("no result in stream");
  return finalData;
}

async function doAnalyze(){
  if(!selectedFile){ setError("No image", "Drop an image or pick an example."); return; }
  const req = requestInput.value.trim() || "count colonies";
  showState("loading"); loadingText.textContent="Analysing…";
  clearLive();
  runBtn.disabled=true; setLeftStatus("");
  const fd=new FormData();
  fd.append("image", selectedFile, selectedFile._name||"image.jpg");
  fd.append("request", req);
  try{
    let data;
    try{
      data=await doAnalyzeStream(fd);
    } catch(_){
      // fallback to non-stream
      const r=await fetch("/api/analyze", {method:"POST", body:fd});
      if(!r.ok){
        let j={}; try{ j=await r.json(); }catch(_){ j={error:await r.text()}; }
        const detail=j.detail||j;
        throw new Error(detail.error||detail.detail||("HTTP "+r.status));
      }
      data=await r.json();
    }
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

function renderResult(data){
  const isT=isTNTC(data.status, data.count);
  countBig.textContent = data.count_label || (isT ? "TNTC >300" : data.count+" CFU");
  countSub.textContent = (data.image_w&&data.image_h)? `${data.image_w}×${data.image_h} · ${data.duration_ms||""} ms` + (data.experimental?" · experimental — not benchmarked":"") : (data.experimental?"experimental — not benchmarked":"");
  const s=(data.status||"ok").toLowerCase();
  statusPill.textContent = s==="tntc" ? "TNTC — dilute & re-plate" : s==="re-photo" ? "Re-photo — retake" : s==="empty" ? "Empty" : "OK";
  statusPill.className="pill "+pillClass(data.status);
  loadViewerImages(data);
  renderTrace(data.trace||[]);
  renderMeasurements(data.measurements||[], data.csv_url);
  renderColonyTypes(data.colony_types||[]);
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
    const parts=[];
    if(t.blur_score!==undefined) parts.push("blur "+t.blur_score);
    if(t.glare_fraction!==undefined) parts.push("glare "+t.glare_fraction);
    if(t.prompt_used) parts.push("prompt "+t.prompt_used);
    if(t.retry!==undefined) parts.push("retry "+(t.retry?"yes":"no"));
    if(t.plate_diameter_px) parts.push("plate "+t.plate_diameter_px+"px");
    if(t.mm_per_px) parts.push(t.mm_per_px+" mm/px");
    if(t.boxes!==undefined) parts.push(t.boxes+" boxes");
    if(t.mean_confidence!=null) parts.push(`conf ${Number(t.mean_confidence).toFixed(2)}${t.low_confidence?" (low)":""}`);
    if(t.count!==undefined) parts.push("count "+t.count);
    if(t.duration_ms!==undefined) parts.push(t.duration_ms+"ms");
    if(t.model) parts.push(t.model);
    if(t.error) parts.push("error: "+t.error);
    kv.textContent=parts.join(" · ")||JSON.stringify(t);
    body.appendChild(name); body.appendChild(kv);
    li.appendChild(icon); li.appendChild(body);
    if(!window.matchMedia("(prefers-reduced-motion: reduce)").matches){
      li.style.opacity="0"; li.style.transform="translateY(6px)"; li.style.transition="opacity .25s, transform .25s";
      setTimeout(()=>{ li.style.opacity="1"; li.style.transform="none"; }, i*80);
    }
    traceList.appendChild(li);
  }
}

function renderMeasurements(rows, csvUrl){
  const tbody=measureTable.querySelector("tbody"); tbody.textContent="";
  if(!rows.length){ const tr=document.createElement("tr"); const td=document.createElement("td"); td.colSpan=5; td.className="muted"; td.textContent="No measurements"; tr.appendChild(td); tbody.appendChild(tr); csvLink.hidden=true; return; }
  for(const r of rows){
    const tr=document.createElement("tr");
    const td0=document.createElement("td"); td0.textContent=String(r.id ?? ""); tr.appendChild(td0);
    const td1=document.createElement("td"); td1.textContent=String(r.area_px ?? r.area ?? ""); tr.appendChild(td1);
    const td2=document.createElement("td"); td2.textContent=String(r.diameter_px ?? r.diameter ?? ""); tr.appendChild(td2);
    const td3=document.createElement("td"); td3.textContent=String(r.area_mm2 ?? ""); tr.appendChild(td3);
    const td4=document.createElement("td"); td4.textContent=String(r.diameter_mm ?? ""); tr.appendChild(td4);
    tbody.appendChild(tr);
  }
  if(csvUrl){ csvLink.href=csvUrl; csvLink.hidden=false; } else csvLink.hidden=true;
}

function renderColonyTypes(types){
  colonyTypesEl.textContent="";
  if(!types.length){ const p=document.createElement("p"); p.className="muted"; p.textContent='No colony types — add "characterize" to your request (e.g. "count and characterize colonies").'; colonyTypesEl.appendChild(p); return; }
  const wrap=document.createElement("div"); wrap.className="table-wrap";
  const table=document.createElement("table");
  const thead=document.createElement("thead"); const trh=document.createElement("tr");
  ["Type","Description","Count","Mean Ø (px)"].forEach(txt=>{ const th=document.createElement("th"); th.textContent=txt; trh.appendChild(th); });
  thead.appendChild(trh); table.appendChild(thead);
  const tbody=document.createElement("tbody");
  for(const t of types){
    const tr=document.createElement("tr");
    const td0=document.createElement("td"); td0.textContent=String(t.label || t.type || ""); tr.appendChild(td0);
    const td1=document.createElement("td"); td1.textContent=String(t.description || ""); tr.appendChild(td1);
    const td2=document.createElement("td"); td2.textContent=String(t.count ?? ""); tr.appendChild(td2);
    const td3=document.createElement("td"); td3.textContent=String(t.mean_diameter_px ?? t.mean_diameter ?? ""); tr.appendChild(td3);
    tbody.appendChild(tr);
  }
  table.appendChild(tbody); wrap.appendChild(table); colonyTypesEl.appendChild(wrap);
}

function loadViewerImages(data){
  const imgUrl = data.image_url;
  const masksUrl = data.masks_url;
  imageEl = new Image(); imageEl.crossOrigin="anonymous";
  masksEl = null;
  let imgLoaded=false;
  function tryDraw(){ if(imgLoaded) drawViewer(); }
  imageEl.onload=()=>{ imgLoaded=true; tryDraw(); };
  imageEl.onerror=()=>{ imgLoaded=true; tryDraw(); };
  imageEl.src = imgUrl;
  if(masksUrl && masksToggle.checked){
    masksEl = new Image(); masksEl.crossOrigin="anonymous";
    masksEl.onload=tryDraw;
    masksEl.onerror=tryDraw;
    masksEl.src = masksUrl;
  } else {
    tryDraw();
  }
  if(data.boxes) lastResult.boxes = data.boxes;
  if(data.image_w) lastResult.image_w = data.image_w;
  if(data.image_h) lastResult.image_h = data.image_h;
}

function drawViewer(){
  if(!lastResult || !imageEl) return;
  const w = lastResult.image_w || imageEl.naturalWidth;
  const h = lastResult.image_h || imageEl.naturalHeight;
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
    for(let i=0;i<lastResult.boxes.length;i++){
      const b=lastResult.boxes[i];
      if(hoveredIdx===i && correctMode==="remove"){
        ctx.strokeStyle="#C8892B"; ctx.lineWidth=Math.max(2, Math.round(w/500));
        ctx.fillStyle="rgba(200,137,43,0.15)";
        ctx.fillRect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
      } else {
        ctx.strokeStyle="#1F6F6A"; ctx.lineWidth=Math.max(1, Math.round(w/700));
      }
      ctx.strokeRect(b.x1, b.y1, b.x2-b.x1, b.y2-b.y1);
    }
  }
  if(lastResult._lastClick){
    const {x,y}=lastResult._lastClick;
    ctx.fillStyle="#C8892B"; ctx.beginPath(); ctx.arc(x,y, Math.max(4, w/250),0,Math.PI*2); ctx.fill();
    ctx.strokeStyle="#fff"; ctx.lineWidth=1.5; ctx.stroke();
  }
}
masksToggle.addEventListener("change", drawViewer);
boxesToggle.addEventListener("change", drawViewer);
window.addEventListener("resize", ()=>{ if(lastResult) drawViewer(); });

function showToast(msg){
  const t=document.getElementById("toast");
  if(!t) return;
  t.textContent=msg; t.hidden=false;
  clearTimeout(t._timer);
  t._timer=setTimeout(()=>{ t.hidden=true; }, 4000);
}

// cursor mode
function updateCursorMode(){
  const wrap=document.getElementById("canvasWrap");
  if(!wrap) return;
  wrap.classList.remove("mode-add","mode-remove","mode-manual");
  wrap.classList.add(correctMode==="add"?"mode-add":correctMode==="manual"?"mode-manual":"mode-remove");
}
updateCursorMode();

function getImageCoords(clientX, clientY){
  const rect = viewer.getBoundingClientRect();
  const w = lastResult.image_w, h = lastResult.image_h;
  const x = (clientX - rect.left) * w / rect.width;
  const y = (clientY - rect.top) * h / rect.height;
  return {x: Math.floor(Math.max(0, Math.min(w-1, x))), y: Math.floor(Math.max(0, Math.min(h-1, y))), rect};
}
let isDragging=false, dragStart=null, dragCurrent=null, hoveredIdx=null;
const dragOverlay=document.getElementById("dragOverlay");
function drawDragRect(){
  if(!dragOverlay || !dragStart || !dragCurrent || !lastResult) return;
  const rect=viewer.getBoundingClientRect();
  const scaleX=rect.width / lastResult.image_w, scaleY=rect.height / lastResult.image_h;
  // convert image coords to canvas display coords
  const x1=Math.min(dragStart.x, dragCurrent.x)*scaleX, y1=Math.min(dragStart.y, dragCurrent.y)*scaleY;
  const x2=Math.max(dragStart.x, dragCurrent.x)*scaleX, y2=Math.max(dragStart.y, dragCurrent.y)*scaleY;
  const c=dragOverlay.getContext("2d");
  dragOverlay.width=rect.width; dragOverlay.height=rect.height;
  c.clearRect(0,0,rect.width,rect.height);
  c.strokeStyle="#1F6F6A"; c.lineWidth=2; c.setLineDash([6,4]);
  c.strokeRect(x1,y1,x2-x1,y2-y1);
  c.fillStyle="rgba(31,111,106,0.15)"; c.fillRect(x1,y1,x2-x1,y2-y1);
}
function clearDragRect(){
  if(!dragOverlay) return;
  const c=dragOverlay.getContext("2d");
  c.clearRect(0,0,dragOverlay.width,dragOverlay.height);
}
function findBoxAtImage(x,y){
  if(!lastResult || !lastResult.boxes) return null;
  for(let i=0;i<lastResult.boxes.length;i++){
    const b=lastResult.boxes[i];
    if(b.x1<=x && x<=b.x2 && b.y1<=y && y<=b.y2) return i;
  }
  return null;
}
viewer.addEventListener("mousedown", (e)=>{
  if(!lastResult || !lastResult.run_id) return;
  if(correctMode!=="add" && correctMode!=="manual") return;
  const p=getImageCoords(e.clientX, e.clientY);
  isDragging=true; dragStart={x:p.x,y:p.y}; dragCurrent={x:p.x,y:p.y};
});
viewer.addEventListener("mousemove", (e)=>{
  if(!lastResult || !lastResult.run_id) return;
  const p=getImageCoords(e.clientX, e.clientY);
  if(isDragging && (correctMode==="add" || correctMode==="manual")){
    dragCurrent={x:p.x,y:p.y};
    drawDragRect();
  } else if(correctMode==="remove"){
    const idx=findBoxAtImage(p.x,p.y);
    if(idx!==hoveredIdx){
      hoveredIdx=idx;
      drawViewer();
    }
  }
});
viewer.addEventListener("mouseup", async (e)=>{
  if(!lastResult || !lastResult.run_id) return;
  if((correctMode==="add" || correctMode==="manual") && isDragging){
    isDragging=false;
    const p=getImageCoords(e.clientX, e.clientY);
    dragCurrent={x:p.x,y:p.y};
    clearDragRect();
    const x1=Math.min(dragStart.x, dragCurrent.x), y1=Math.min(dragStart.y, dragCurrent.y);
    const x2=Math.max(dragStart.x, dragCurrent.x), y2=Math.max(dragStart.y, dragCurrent.y);
    const dist=Math.max(Math.abs(x2-x1), Math.abs(y2-y1));
    let payload;
    const modeToSend = correctMode==="manual" ? "manual" : "add";
    if(dist < 6){
      payload={run_id:lastResult.run_id, x:Math.floor((x1+x2)/2), y:Math.floor((y1+y2)/2), mode:modeToSend};
    } else {
      // clamp inclusive
      const w=lastResult.image_w, h=lastResult.image_h;
      payload={run_id:lastResult.run_id, x1:Math.max(0,Math.min(w-1,x1)), y1:Math.max(0,Math.min(h-1,y1)), x2:Math.max(0,Math.min(w-1,x2)), y2:Math.max(0,Math.min(h-1,y2)), mode:modeToSend};
    }
    dragStart=null; dragCurrent=null;
    lastResult._lastClick={x:payload.x||payload.x1, y:payload.y||payload.y1}; drawViewer();
    clickSpinner.hidden=false;
    try{
      const r=await fetch("/api/correct", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      if(!r.ok){
        let j={}; try{ j=await r.json(); }catch(_){}
        const msg=(j.detail&&j.detail.error)||j.error||("HTTP "+r.status);
        const hint=(j.detail&&j.detail.hint)||j.hint||"";
        throw new Error(msg + (hint?" — "+hint:""));
      }
      const data=await r.json();
      lastResult.count = data.count; lastResult.count_label = data.count_label; lastResult.boxes = data.boxes;
      if(data.status) lastResult.status = data.status;
      if(data.measurements) lastResult.measurements = data.measurements;
      if(data.correction) lastResult._lastCorrection = data.correction;
      countBig.textContent = data.count_label || data.count+" CFU";
      statusPill.textContent = (data.status==="tntc"?"TNTC — dilute & re-plate":data.status||"ok");
      statusPill.className="pill "+pillClass(data.status);
      renderMeasurements(data.measurements||[], data.csv_url||lastResult.csv_url);
      if(lastResult.masks_url) masksEl=null;
      const bust = "?t="+Date.now();
      if(masksToggle.checked){
        masksEl = new Image(); masksEl.crossOrigin="anonymous";
        masksEl.onload=drawViewer; masksEl.src = lastResult.masks_url + bust;
      }
      drawViewer();
      showToast(modeToSend==="manual" ? "Manually marked — count updated (not measured)" : "Added colony — count updated");
    }catch(err){
      const msg=err.message.slice(0,200);
      setLeftStatus(msg); showToast(msg);
    }finally{ clickSpinner.hidden=true; }
    return;
  }
  if(correctMode==="remove" && !isDragging){
    // click to remove
    const {x,y}=getImageCoords(e.clientX, e.clientY);
    lastResult._lastClick={x,y}; drawViewer();
    clickSpinner.hidden=false;
    try{
      const r=await fetch("/api/correct", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({run_id:lastResult.run_id, x,y, mode:"remove"})});
      if(!r.ok){
        let j={}; try{ j=await r.json(); }catch(_){}
        const msg=(j.detail&&j.detail.error)||j.error||("HTTP "+r.status);
        const hint=(j.detail&&j.detail.hint)||j.hint||"";
        throw new Error(msg + (hint?" — "+hint:""));
      }
      const data=await r.json();
      lastResult.count = data.count; lastResult.count_label = data.count_label; lastResult.boxes = data.boxes;
      if(data.status) lastResult.status = data.status;
      if(data.measurements) lastResult.measurements = data.measurements;
      if(data.correction) lastResult._lastCorrection = data.correction;
      countBig.textContent = data.count_label || data.count+" CFU";
      statusPill.textContent = (data.status==="tntc"?"TNTC — dilute & re-plate":data.status||"ok");
      statusPill.className="pill "+pillClass(data.status);
      renderMeasurements(data.measurements||[], data.csv_url||lastResult.csv_url);
      if(lastResult.masks_url) masksEl=null;
      const bust = "?t="+Date.now();
      if(masksToggle.checked){
        masksEl = new Image(); masksEl.crossOrigin="anonymous";
        masksEl.onload=drawViewer; masksEl.src = lastResult.masks_url + bust;
      }
      drawViewer();
      showToast("Removed colony — count updated");
    }catch(err){
      const msg=err.message.slice(0,200);
      setLeftStatus(msg); showToast(msg);
    }finally{ clickSpinner.hidden=true; }
  }
  isDragging=false; clearDragRect();
});
viewer.addEventListener("mouseleave", ()=>{ isDragging=false; clearDragRect(); hoveredIdx=null; drawViewer(); });

const modeView=document.getElementById("modeView");
if(modeView){ modeView.addEventListener("click", ()=>{ correctMode="view"; document.querySelectorAll(".seg-control .btn").forEach(b=>b.classList.remove("is-active")); modeView.classList.add("is-active"); updateCursorMode(); hoveredIdx=null; drawViewer(); }); }
modeAdd.addEventListener("click", ()=>{ correctMode="add"; document.querySelectorAll(".seg-control .btn").forEach(b=>b.classList.remove("is-active")); modeAdd.classList.add("is-active"); updateCursorMode(); hoveredIdx=null; drawViewer(); });
modeRemove.addEventListener("click", ()=>{ correctMode="remove"; document.querySelectorAll(".seg-control .btn").forEach(b=>b.classList.remove("is-active")); modeRemove.classList.add("is-active"); updateCursorMode(); hoveredIdx=null; drawViewer(); });
if(modeManual){
  modeManual.addEventListener("click", ()=>{ correctMode="manual"; document.querySelectorAll(".seg-control .btn").forEach(b=>b.classList.remove("is-active")); modeManual.classList.add("is-active"); updateCursorMode(); hoveredIdx=null; drawViewer(); });
}

document.querySelectorAll(".tab").forEach(btn=>{
  btn.addEventListener("click", ()=>{
    document.querySelectorAll(".tab").forEach(b=>{ b.classList.remove("is-active"); b.setAttribute("aria-selected","false"); });
    btn.classList.add("is-active"); btn.setAttribute("aria-selected","true");
    const t=btn.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach(p=> p.hidden=true);
    $("#tab-"+t).hidden=false;
  });
});

// mic: Web Audio API -> resample to 24kHz mono -> 16-bit PCM WAV -> /api/transcribe
let recState=null;
function encodeWAV(pcmFloat32, sampleRate){
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

requestInput.addEventListener("keydown", e=>{ if(e.key==="Enter" && (e.metaKey||e.ctrlKey)){ doAnalyze(); }});
