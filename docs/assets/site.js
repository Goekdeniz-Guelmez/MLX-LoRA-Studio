(function () {
  var observer;
  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function enhance() {
    var items = document.querySelectorAll(
      ".hero__copy, .hero__visual, .home-section, .principle, .workflow, .audience, .method-path, .quote-panel, .showcase, .markdown-section > h1, .markdown-section > h2, .markdown-section > h3"
    );

    if (observer) observer.disconnect();
    if (reducedMotion || !("IntersectionObserver" in window)) {
      items.forEach(function (item) { item.classList.add("is-visible"); });
      return;
    }

    observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        } else {
          entry.target.classList.remove("is-visible");
        }
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -35px 0px" });

    items.forEach(function (item) {
      item.classList.add("reveal-ready");
      observer.observe(item);
    });
  }

  window.MLXStudioSite = { enhance: enhance };
})();
