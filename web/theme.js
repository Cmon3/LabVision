// Theme: "system" (follow the OS), "light" or "dark". Loaded in <head> so the saved
// choice is applied before the page paints (no flash of the wrong theme).
(function () {
  var KEY = "labvision_theme";
  var MODES = ["system", "light", "dark"];
  var LABEL = { system: "System", light: "Light", dark: "Dark" };
  var ICON = { system: "◐", light: "☀", dark: "☾" };

  function read() {
    try { var v = localStorage.getItem(KEY); return MODES.indexOf(v) >= 0 ? v : "system"; }
    catch (_) { return "system"; }
  }
  function apply(mode) {
    var root = document.documentElement;
    if (mode === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", mode);
  }
  apply(read());

  function mountToggle() {
    var nav = document.querySelector("header nav");
    if (!nav || document.getElementById("themeToggle")) return;
    var btn = document.createElement("button");
    btn.id = "themeToggle";
    btn.type = "button";
    btn.className = "theme-toggle";
    function render(mode) {
      btn.textContent = ICON[mode] + " " + LABEL[mode];
      btn.setAttribute("aria-label", "Theme: " + LABEL[mode] + ". Click to change.");
      btn.title = "Theme: " + LABEL[mode] + " (System → Light → Dark)";
    }
    render(read());
    btn.addEventListener("click", function () {
      var next = MODES[(MODES.indexOf(read()) + 1) % MODES.length];
      try { localStorage.setItem(KEY, next); } catch (_) {}
      apply(next);
      render(next);
    });
    nav.appendChild(btn);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", mountToggle);
  else mountToggle();
})();
