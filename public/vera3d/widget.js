// Ask-Vera embed: one line on any site.
//   <script src="https://verantyx.ai/vera3d/widget.js" data-q="時効とは"></script>
(function(){
  var me = document.currentScript;
  var q = me.getAttribute("data-q") || "";
  var f = document.createElement("iframe");
  f.src = "https://verantyx.ai/vera3d/?embed=1" +
          (q ? "&q=" + encodeURIComponent(q) : "");
  f.style.cssText = "width:100%;max-width:520px;height:560px;border:1px solid #333;border-radius:8px";
  f.loading = "lazy";
  me.parentNode.insertBefore(f, me);
})();
