// Batch upload shared by /app and /app/queue.
// Keeps only images (a picked folder may contain notes, .DS_Store, sub-folders' files...),
// then sends them to POST /api/batch in requests of at most 50 (the server's per-request limit).
// The server analyses them in the background, one at a time, and they appear in the queue.
(function () {
  var CHUNK = 50;
  var IMG = /\.(jpe?g|png|tiff?)$/i;

  function isImage(f) {
    return (f.type && /^image\/(jpeg|png|tiff)$/.test(f.type)) || IMG.test(f.name || "");
  }

  async function uploadBatch(fileList, onProgress) {
    var all = Array.prototype.slice.call(fileList || []);
    var files = all.filter(function (f) { return isImage(f) && !(f.name || "").startsWith("."); });
    var result = { accepted: [], rejected: [], ignored: all.length - files.length };
    for (var i = 0; i < files.length; i += CHUNK) {
      var part = files.slice(i, i + CHUNK);
      if (onProgress) onProgress(Math.min(i + part.length, files.length), files.length);
      var fd = new FormData();
      part.forEach(function (f) { fd.append("images", f, f.name); });
      var r = await fetch("/api/batch", { method: "POST", body: fd });
      var j = await r.json().catch(function () { return {}; });
      if (!r.ok) throw new Error((j.detail && j.detail.hint) || ("HTTP " + r.status));
      result.accepted = result.accepted.concat(j.accepted || []);
      result.rejected = result.rejected.concat(j.rejected || []);
    }
    return result;
  }

  window.LabVisionBatch = { uploadBatch: uploadBatch, isImage: isImage };
})();
