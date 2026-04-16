import os
import shutil
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Query, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from app.video_processing import extract_frames
from app.embeddings import get_image_embedding, get_text_embedding
from app.search import (
    add_embedding, search, load_index, save_index,
    is_video_processed, mark_video_processed, clear_index,
    get_indexed_videos, remove_video, remove_folder,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_index()
    yield
    save_index()


app = FastAPI(title="Video Search Engine", lifespan=lifespan)

UPLOAD_FOLDER = "data/videos"
FRAME_FOLDER = "data/frames"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAME_FOLDER, exist_ok=True)

processing_state = {"active": False, "current": "", "done": 0, "total": 0, "errors": []}


# ── Serve files ───────────────────────────────────────────────
@app.get("/video")
def serve_video(path: str):
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/frame")
def serve_frame(path: str):
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# ── Upload multiple videos ───────────────────────────────────
@app.post("/upload")
async def upload_videos(files: list[UploadFile] = File(...)):
    if processing_state["active"]:
        return JSONResponse({"error": "Processing already in progress"}, status_code=409)
    saved_paths = []
    for file in files:
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(video_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        saved_paths.append(video_path)
    asyncio.get_event_loop().run_in_executor(None, _process_folder_sync, saved_paths)
    return {"message": f"Started processing {len(saved_paths)} video(s)", "total": len(saved_paths)}


# ── Process a local folder ───────────────────────────────────
@app.post("/process-folder")
async def process_folder(request: Request):
    body = await request.json()
    folder = body.get("folder", "").strip()
    if not folder or not os.path.isdir(folder):
        return JSONResponse({"error": f"Folder not found: {folder}"}, status_code=400)
    if processing_state["active"]:
        return JSONResponse({"error": "Processing already in progress"}, status_code=409)
    video_files = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                video_files.append(os.path.join(root, fname))
    if not video_files:
        return JSONResponse({"error": "No video files found in folder"}, status_code=400)
    asyncio.get_event_loop().run_in_executor(None, _process_folder_sync, video_files)
    return {"message": f"Started processing {len(video_files)} videos", "total": len(video_files)}


def _process_folder_sync(video_files: list[str]):
    processing_state.update(active=True, done=0, total=len(video_files), errors=[], current="")
    for vpath in video_files:
        processing_state["current"] = os.path.basename(vpath)
        try:
            if is_video_processed(vpath):
                print(f"Skipping (already indexed): {vpath}")
            else:
                _process_single_video(vpath)
        except Exception as exc:
            processing_state["errors"].append(f"{os.path.basename(vpath)}: {exc}")
            print(f"Error processing {vpath}: {exc}")
        processing_state["done"] += 1
    save_index()
    processing_state["active"] = False
    processing_state["current"] = ""
    print("Folder processing complete")


def _process_single_video(video_path: str) -> dict:
    video_path = os.path.normpath(video_path)
    name = Path(video_path).stem
    frame_folder = os.path.join(FRAME_FOLDER, name)
    frame_paths = extract_frames(video_path, frame_folder)
    embedded = 0
    for i, fpath in enumerate(frame_paths):
        emb = get_image_embedding(fpath)
        if emb is None:
            continue
        add_embedding(emb, video_path, frame_path=fpath, frame_index=i)
        embedded += 1
    mark_video_processed(video_path)
    print(f"Indexed {embedded}/{len(frame_paths)} frames for {video_path}")
    return {"video": video_path, "frames": len(frame_paths), "embedded": embedded}


@app.get("/progress")
def get_progress():
    return processing_state


@app.get("/search")
def search_videos(query: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=50)):
    emb = get_text_embedding(query)
    if emb is None:
        return JSONResponse({"error": "Failed to create text embedding"}, status_code=500)
    results = search(emb, k=k)
    for r in results:
        r["filename"] = os.path.basename(r["video_path"])
    return {"query": query, "results": results}


@app.get("/indexed")
def list_indexed():
    return {"videos": get_indexed_videos()}


@app.post("/clear")
def clear_all():
    clear_index()
    return {"message": "Index cleared"}


@app.post("/remove-video")
async def api_remove_video(request: Request):
    body = await request.json()
    vpath = body.get("video_path", "").strip()
    if not vpath:
        return JSONResponse({"error": "video_path required"}, status_code=400)
    found = remove_video(vpath)
    if found:
        save_index()
        return {"message": f"Removed: {os.path.basename(vpath)}"}
    return JSONResponse({"error": "Video not found in index"}, status_code=404)


@app.post("/remove-folder")
async def api_remove_folder(request: Request):
    body = await request.json()
    folder = body.get("folder", "").strip()
    if not folder:
        return JSONResponse({"error": "folder required"}, status_code=400)
    count = remove_folder(folder)
    if count:
        save_index()
        return {"message": f"Removed {count} video(s) from {folder}"}
    return JSONResponse({"error": "No indexed videos found from that folder"}, status_code=404)


# ── UI ────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def ui():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Video Search Engine</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e2e8f0;min-height:100vh}
.container{max-width:1200px;margin:0 auto;padding:24px}

h1{font-size:2.2rem;font-weight:700;text-align:center;margin-bottom:6px;
   background:linear-gradient(135deg,#6366f1,#a78bfa,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{text-align:center;color:#94a3b8;margin-bottom:28px;font-size:.95rem}

/* ── Navigation Buttons ── */
.nav-bar{display:flex;gap:8px;margin-bottom:28px}
.nav-btn{
  flex:1;padding:14px 10px;text-align:center;border-radius:10px;cursor:pointer;
  font-weight:700;font-size:.95rem;border:2px solid #2a2d3a;background:#1a1d27;color:#94a3b8;
  transition:all .2s;user-select:none;
}
.nav-btn:hover{border-color:#6366f1;color:#e2e8f0}
.nav-btn-active{background:#6366f1;color:#fff;border-color:#6366f1}

/* ── Page Sections ── */
.page{display:none}
.page-visible{display:block}

/* ── Panels ── */
.panel{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;padding:24px;margin-bottom:20px}
.panel-title{font-size:1rem;font-weight:600;margin-bottom:14px;display:flex;align-items:center;gap:8px}
.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
input[type=text],input[type=search]{
  flex:1;min-width:200px;padding:12px 16px;border-radius:8px;border:1px solid #2a2d3a;
  background:#0d0f16;color:#e2e8f0;font-size:.95rem;outline:none;transition:border .2s}
input:focus{border-color:#6366f1}

/* ── Buttons ── */
button{padding:10px 20px;border-radius:8px;border:none;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .15s}
button:active{transform:scale(.97)}
button:disabled{opacity:.5;cursor:not-allowed}
.btn-primary{background:#6366f1;color:#fff}
.btn-primary:hover:not(:disabled){background:#818cf8}
.btn-danger{background:#ef4444;color:#fff}
.btn-danger:hover:not(:disabled){background:#dc2626}
.btn-sm{padding:6px 14px;font-size:.8rem;border-radius:6px}
.btn-outline{background:transparent;border:1px solid #2a2d3a;color:#94a3b8}
.btn-outline:hover{border-color:#ef4444;color:#ef4444}
.hint{font-size:.8rem;color:#94a3b8;margin-top:6px}

/* ── Divider ── */
.divider{display:flex;align-items:center;gap:12px;margin:20px 0;color:#94a3b8;font-size:.85rem;font-weight:600}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:#2a2d3a}

/* ── Progress ── */
.progress-wrap{margin-top:16px;display:none}
.progress-bar-bg{height:8px;background:#2a2d3a;border-radius:4px;overflow:hidden}
.progress-bar{height:100%;background:linear-gradient(90deg,#6366f1,#06b6d4);border-radius:4px;transition:width .4s}
.progress-text{font-size:.85rem;color:#94a3b8;margin-top:6px}

/* ── Status Messages ── */
.status-msg{margin-top:12px;padding:12px 16px;border-radius:8px;font-size:.9rem;display:none}
.status-ok{display:block;background:rgba(34,197,94,.1);color:#22c55e;border:1px solid rgba(34,197,94,.2)}
.status-err{display:block;background:rgba(239,68,68,.1);color:#ef4444;border:1px solid rgba(239,68,68,.2)}
.status-info{display:block;background:rgba(99,102,241,.1);color:#818cf8;border:1px solid rgba(99,102,241,.2)}

/* ── Search Box ── */
.search-wrap{position:relative}
.search-wrap input{width:100%;padding:16px 120px 16px 20px;font-size:1.1rem;border-radius:12px}
.search-wrap button{position:absolute;right:8px;top:50%;transform:translateY(-50%);padding:10px 24px}

/* ── Results Grid ── */
.results-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:20px;margin-top:20px}
.result-card{background:#1a1d27;border:1px solid #2a2d3a;border-radius:12px;overflow:hidden;transition:all .2s}
.result-card:hover{border-color:#6366f1;box-shadow:0 0 24px rgba(99,102,241,.12)}
.result-card video{width:100%;aspect-ratio:16/9;object-fit:cover;background:#000;display:block}
.result-info{padding:14px 16px}
.result-title{font-weight:600;font-size:.95rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.result-meta{display:flex;justify-content:space-between;align-items:center;margin-top:6px}
.score-bar-bg{flex:1;height:6px;background:#2a2d3a;border-radius:3px;margin-right:10px}
.score-bar{height:100%;border-radius:3px;transition:width .3s}
.score-label{font-size:.8rem;font-weight:700;min-width:44px;text-align:right}

/* ── Index List ── */
.idx-list{max-height:400px;overflow-y:auto;margin-top:12px}
.idx-item{display:flex;align-items:center;justify-content:space-between;padding:10px 14px;
          border:1px solid #2a2d3a;border-radius:8px;margin-bottom:6px;transition:background .15s}
.idx-item:hover{background:rgba(99,102,241,.05)}
.idx-name{flex:1;font-size:.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-right:12px}
.idx-frames{font-size:.8rem;color:#94a3b8;margin-right:12px;white-space:nowrap}
.idx-empty{text-align:center;padding:40px 20px;color:#94a3b8}

.empty-state{text-align:center;padding:60px 20px;color:#94a3b8}
.empty-state .icon{font-size:3rem;margin-bottom:12px}
.toolbar{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.result-count{font-size:.9rem;color:#94a3b8}

@media(max-width:640px){
  .results-grid{grid-template-columns:1fr}
  .row{flex-direction:column}
  .row>*{width:100%}
  .nav-bar{flex-direction:column}
}
</style>
</head>
<body>
<div class="container">
  <h1>Video Search Engine</h1>
  <p class="subtitle">Index your videos, then search them with natural language</p>

  <!-- ====== NAVIGATION BUTTONS ====== -->
  <div class="nav-bar">
    <button class="nav-btn nav-btn-active" id="navSearch"  onclick="showPage('search')">Search Videos</button>
    <button class="nav-btn"                 id="navIndex"   onclick="showPage('index')">Index Videos</button>
    <button class="nav-btn"                 id="navManage"  onclick="showPage('manage')">Manage Index</button>
  </div>

  <!-- ====== PAGE: SEARCH ====== -->
  <div class="page page-visible" id="pageSearch">
    <div class="panel">
      <div class="search-wrap">
        <input type="search" id="searchBox" placeholder="Describe what you are looking for..." onkeydown="if(event.key==='Enter')doSearch()" />
        <button class="btn-primary" onclick="doSearch()">Search</button>
      </div>
    </div>
    <div id="resultsArea">
      <div class="empty-state">
        <div class="icon">&#128269;</div>
        <p>Index some videos first, then search here!</p>
      </div>
    </div>
  </div>

  <!-- ====== PAGE: INDEX ====== -->
  <div class="page" id="pageIndex">

    <!-- Option A: Folder path -->
    <div class="panel">
      <div class="panel-title">Index from Folder Path</div>
      <div class="row">
        <input type="text" id="folderPath" placeholder="Paste full folder path  e.g.  D:\MyVideos" />
        <button class="btn-primary" id="folderBtn" onclick="processFolder()">Process Folder</button>
      </div>
      <p class="hint">Recursively scans for .mp4 .avi .mkv .mov .wmv .flv .webm .m4v</p>
      <div class="status-msg" id="folderStatus"></div>
    </div>

    <div class="divider">OR</div>

    <!-- Option B: Browse folder -->
    <div class="panel">
      <div class="panel-title">Browse &amp; Select a Folder</div>
      <div class="row">
        <input type="file" id="folderInput" webkitdirectory multiple style="flex:1;color:#94a3b8" />
        <button class="btn-primary" id="browseFolderBtn" onclick="uploadFolderFiles()">Upload Folder</button>
      </div>
      <p class="hint">Select a folder — all videos inside will be uploaded &amp; indexed</p>
      <div class="status-msg" id="browseFolderStatus"></div>
    </div>

    <div class="divider">OR</div>

    <!-- Option C: Individual files -->
    <div class="panel">
      <div class="panel-title">Upload Individual Videos</div>
      <div class="row">
        <input type="file" id="videoFiles" accept="video/*" multiple style="flex:1;color:#94a3b8" />
        <button class="btn-primary" id="uploadBtn" onclick="uploadVideos()">Upload &amp; Index</button>
      </div>
      <p class="hint">Hold Ctrl / Shift to select multiple files</p>
      <div class="status-msg" id="uploadStatus"></div>
    </div>

    <!-- Shared progress bar -->
    <div class="progress-wrap" id="progressWrap">
      <div class="progress-bar-bg"><div class="progress-bar" id="progressBar" style="width:0%"></div></div>
      <p class="progress-text" id="progressText">Starting...</p>
    </div>
  </div>

  <!-- ====== PAGE: MANAGE ====== -->
  <div class="page" id="pageManage">
    <div class="panel">
      <div class="panel-title" style="justify-content:space-between;width:100%">
        <span>Indexed Videos</span>
        <button class="btn-sm btn-primary" onclick="loadIndexed()">Refresh</button>
      </div>

      <!-- Remove by folder -->
      <div class="row" style="margin-bottom:12px">
        <input type="text" id="removeFolderPath" placeholder="Folder path to remove all its videos..." style="font-size:.85rem;padding:10px 14px" />
        <button class="btn-sm btn-danger" onclick="doRemoveFolder()">Remove Folder</button>
      </div>
      <div class="status-msg" id="manageStatus"></div>

      <div class="idx-list" id="indexedList">
        <div class="idx-empty">Click "Refresh" to load indexed videos</div>
      </div>

      <div style="margin-top:16px;text-align:right">
        <button class="btn-danger" onclick="clearAll()">Clear Entire Index</button>
      </div>
    </div>
  </div>

</div>

<script>
var API='';

/* ═══════════════════════════════════════════════
   PAGE SWITCHING — simple show/hide
   ═══════════════════════════════════════════════ */
var pages  = {search:'pageSearch', index:'pageIndex', manage:'pageManage'};
var navs   = {search:'navSearch',  index:'navIndex',  manage:'navManage'};

function showPage(name){
  // Hide all pages
  var keys = ['search','index','manage'];
  for(var i=0;i<keys.length;i++){
    document.getElementById(pages[keys[i]]).style.display = 'none';
    document.getElementById(navs[keys[i]]).className = 'nav-btn';
  }
  // Show selected
  document.getElementById(pages[name]).style.display = 'block';
  document.getElementById(navs[name]).className  = 'nav-btn nav-btn-active';

  if(name==='manage') loadIndexed();
}

/* ═══════════════════════════════════════════════
   FOLDER PATH PROCESSING
   ═══════════════════════════════════════════════ */
function processFolder(){
  var folder=document.getElementById('folderPath').value.trim();
  if(!folder){setStatus('folderStatus','Enter a folder path','err');return;}
  var btn=document.getElementById('folderBtn');
  btn.disabled=true;btn.textContent='Processing...';
  setStatus('folderStatus','','');
  fetch(API+'/process-folder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:folder})})
  .then(function(r){return r.json().then(function(d){return{ok:r.ok,data:d}})})
  .then(function(r){
    if(!r.ok){setStatus('folderStatus',r.data.error||'Error','err');btn.disabled=false;btn.textContent='Process Folder';return;}
    setStatus('folderStatus',r.data.message,'info');
    document.getElementById('progressWrap').style.display='block';
    pollProgress('folderStatus','folderBtn','Process Folder');
  })
  .catch(function(e){setStatus('folderStatus','Network error: '+e,'err');btn.disabled=false;btn.textContent='Process Folder';});
}

/* ═══════════════════════════════════════════════
   BROWSE FOLDER UPLOAD
   ═══════════════════════════════════════════════ */
function uploadFolderFiles(){
  var input=document.getElementById('folderInput');
  var allFiles=Array.from(input.files);
  var exts=['.mp4','.avi','.mkv','.mov','.wmv','.flv','.webm','.m4v'];
  var videos=allFiles.filter(function(f){
    var n=f.name.toLowerCase();
    return exts.some(function(e){return n.endsWith(e);});
  });
  if(!videos.length){setStatus('browseFolderStatus','No video files found in selected folder','err');return;}

  var btn=document.getElementById('browseFolderBtn');
  btn.disabled=true;btn.textContent='Uploading...';
  var form=new FormData();
  for(var i=0;i<videos.length;i++) form.append('files',videos[i]);
  setStatus('browseFolderStatus','Uploading '+videos.length+' video(s)...','info');

  fetch(API+'/upload',{method:'POST',body:form})
  .then(function(r){return r.json().then(function(d){return{ok:r.ok,data:d}})})
  .then(function(r){
    if(!r.ok){setStatus('browseFolderStatus',r.data.error||'Failed','err');btn.disabled=false;btn.textContent='Upload Folder';return;}
    setStatus('browseFolderStatus',r.data.message,'info');
    document.getElementById('progressWrap').style.display='block';
    pollProgress('browseFolderStatus','browseFolderBtn','Upload Folder');
  })
  .catch(function(e){setStatus('browseFolderStatus','Error: '+e,'err');btn.disabled=false;btn.textContent='Upload Folder';});
}

/* ═══════════════════════════════════════════════
   UPLOAD INDIVIDUAL VIDEOS
   ═══════════════════════════════════════════════ */
function uploadVideos(){
  var input=document.getElementById('videoFiles');
  if(!input.files.length){setStatus('uploadStatus','Select one or more video files','err');return;}
  var btn=document.getElementById('uploadBtn');
  btn.disabled=true;btn.textContent='Uploading...';
  var form=new FormData();
  for(var i=0;i<input.files.length;i++) form.append('files',input.files[i]);
  setStatus('uploadStatus','Uploading '+input.files.length+' video(s)...','info');

  fetch(API+'/upload',{method:'POST',body:form})
  .then(function(r){return r.json().then(function(d){return{ok:r.ok,data:d}})})
  .then(function(r){
    if(!r.ok){setStatus('uploadStatus',r.data.error||'Failed','err');btn.disabled=false;btn.textContent='Upload & Index';return;}
    setStatus('uploadStatus',r.data.message,'info');
    document.getElementById('progressWrap').style.display='block';
    pollProgress('uploadStatus','uploadBtn','Upload & Index');
  })
  .catch(function(e){setStatus('uploadStatus','Error: '+e,'err');btn.disabled=false;btn.textContent='Upload & Index';});
}

/* ═══════════════════════════════════════════════
   PROGRESS POLLING
   ═══════════════════════════════════════════════ */
var pollTimer=null;
function pollProgress(statusId,btnId,btnLabel){
  if(pollTimer){clearInterval(pollTimer);pollTimer=null;}
  pollTimer=setInterval(function(){
    fetch(API+'/progress').then(function(r){return r.json()}).then(function(d){
      var pct=d.total?Math.round(d.done/d.total*100):0;
      document.getElementById('progressBar').style.width=pct+'%';
      if(d.active){
        document.getElementById('progressText').textContent='Processing: '+d.current+' ('+d.done+'/'+d.total+')';
      }else{
        document.getElementById('progressText').textContent='Done! '+d.done+'/'+d.total+' videos indexed';
        clearInterval(pollTimer);pollTimer=null;
        document.getElementById(btnId).disabled=false;
        document.getElementById(btnId).textContent=btnLabel;
        var errTxt=(d.errors&&d.errors.length)?' | Errors: '+d.errors.join(', '):'';
        setStatus(statusId,'Indexing complete! '+d.done+' videos processed.'+errTxt,'ok');
      }
    }).catch(function(){});
  },1500);
}

/* ═══════════════════════════════════════════════
   SEARCH
   ═══════════════════════════════════════════════ */
function doSearch(){
  var q=document.getElementById('searchBox').value.trim();
  if(!q) return;
  var area=document.getElementById('resultsArea');
  area.innerHTML='<p style="text-align:center;color:#94a3b8;padding:40px">Searching...</p>';

  fetch(API+'/search?query='+encodeURIComponent(q)+'&k=12')
  .then(function(r){return r.json()})
  .then(function(data){
    if(!data.results){
      area.innerHTML='<p style="color:#ef4444;text-align:center;padding:40px">'+(data.error||'Search failed')+'</p>';
      return;
    }
    if(data.results.length===0){
      area.innerHTML='<div class="empty-state"><div class="icon">&#129300;</div><p>No results for "'+q+'"</p></div>';
      return;
    }
    var html='<div class="toolbar"><span class="result-count">'+data.results.length+' result(s) for "'+q+'"</span></div>';
    html+='<div class="results-grid">';
    for(var i=0;i<data.results.length;i++){
      var r=data.results[i];
      var vidUrl=API+'/video?path='+encodeURIComponent(r.video_path);
      var thumbUrl=r.matched_frame?(API+'/frame?path='+encodeURIComponent(r.matched_frame)):'';
      var score=r.score;
      var barColor=score>=70?'#22c55e':score>=40?'#f59e0b':'#ef4444';
      var scoreColor=score>=70?'#22c55e':score>=40?'#f59e0b':'#94a3b8';
      html+='<div class="result-card">';
      html+='<video controls preload="metadata" poster="'+thumbUrl+'"><source src="'+vidUrl+'" type="video/mp4"></video>';
      html+='<div class="result-info">';
      html+='<div class="result-title" title="'+r.filename+'">'+r.filename+'</div>';
      html+='<div class="result-meta">';
      html+='<div class="score-bar-bg"><div class="score-bar" style="width:'+score+'%;background:'+barColor+'"></div></div>';
      html+='<div class="score-label" style="color:'+scoreColor+'">'+score+'%</div>';
      html+='</div></div></div>';
    }
    html+='</div>';
    area.innerHTML=html;
  })
  .catch(function(e){area.innerHTML='<p style="color:#ef4444;text-align:center;padding:40px">Error: '+e+'</p>';});
}

/* ═══════════════════════════════════════════════
   MANAGE INDEX
   ═══════════════════════════════════════════════ */
function loadIndexed(){
  var list=document.getElementById('indexedList');
  list.innerHTML='<div class="idx-empty">Loading...</div>';

  fetch(API+'/indexed').then(function(r){return r.json()}).then(function(data){
    if(!data.videos||data.videos.length===0){
      list.innerHTML='<div class="idx-empty">No videos indexed yet.</div>';
      return;
    }
    var html='';
    for(var i=0;i<data.videos.length;i++){
      var v=data.videos[i];
      var name=v.video_path.replace(/\\/g,'/').split('/').pop();
      var safeP=v.video_path.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
      html+='<div class="idx-item">';
      html+='<span class="idx-name" title="'+v.video_path+'">'+name+'</span>';
      html+='<span class="idx-frames">'+v.frames+' frames</span>';
      html+='<button class="btn-sm btn-outline" onclick="doRemoveVideo(\''+safeP+'\')">Remove</button>';
      html+='</div>';
    }
    list.innerHTML=html;
  }).catch(function(){list.innerHTML='<div class="idx-empty">Error loading index</div>';});
}

function doRemoveVideo(vpath){
  if(!confirm('Remove this video from the index?')) return;
  fetch(API+'/remove-video',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({video_path:vpath})})
  .then(function(r){return r.json().then(function(d){return{ok:r.ok,data:d}})})
  .then(function(r){
    setStatus('manageStatus',r.data.message||r.data.error,r.ok?'ok':'err');
    loadIndexed();
  })
  .catch(function(e){setStatus('manageStatus','Error: '+e,'err');});
}

function doRemoveFolder(){
  var folder=document.getElementById('removeFolderPath').value.trim();
  if(!folder){setStatus('manageStatus','Enter a folder path','err');return;}
  if(!confirm('Remove ALL videos from "'+folder+'" from the index?')) return;
  fetch(API+'/remove-folder',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({folder:folder})})
  .then(function(r){return r.json().then(function(d){return{ok:r.ok,data:d}})})
  .then(function(r){
    setStatus('manageStatus',r.data.message||r.data.error,r.ok?'ok':'err');
    loadIndexed();
  })
  .catch(function(e){setStatus('manageStatus','Error: '+e,'err');});
}

function clearAll(){
  if(!confirm('This will DELETE the entire index. Are you sure?')) return;
  fetch(API+'/clear',{method:'POST'}).then(function(){
    setStatus('manageStatus','Index cleared','ok');
    loadIndexed();
  }).catch(function(e){setStatus('manageStatus','Error: '+e,'err');});
}

/* ═══════════════════════════════════════════════
   STATUS HELPER
   ═══════════════════════════════════════════════ */
function setStatus(id,msg,type){
  var el=document.getElementById(id);
  el.className='status-msg';
  if(type==='ok')   el.className='status-msg status-ok';
  if(type==='err')  el.className='status-msg status-err';
  if(type==='info') el.className='status-msg status-info';
  el.textContent=msg||'';
}
</script>
</body>
</html>"""
