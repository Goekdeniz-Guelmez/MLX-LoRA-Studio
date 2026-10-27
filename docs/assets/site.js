(function () {
  var observer;
  var latestDmgPromise;
  var releasesUrl = "https://github.com/Goekdeniz-Guelmez/MLX-LoRA-Studio/releases/latest";
  var latestReleaseApi = "https://api.github.com/repos/Goekdeniz-Guelmez/MLX-LoRA-Studio/releases/latest";
  var reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function getLatestDmg() {
    if (!latestDmgPromise) {
      latestDmgPromise = window.fetch(latestReleaseApi, {
        headers: { Accept: "application/vnd.github+json" }
      }).then(function (response) {
        if (!response.ok) throw new Error("Could not read the latest GitHub release.");
        return response.json();
      }).then(function (release) {
        var asset = (release.assets || []).find(function (item) {
          return typeof item.name === "string" && item.name.toLowerCase().endsWith(".dmg");
        });
        if (!asset || !asset.browser_download_url) throw new Error("The latest release has no DMG asset.");
        return { url: asset.browser_download_url, version: release.tag_name || release.name };
      }).catch(function (error) {
        latestDmgPromise = null;
        throw error;
      });
    }
    return latestDmgPromise;
  }

  function configureLatestDownloads() {
    var links = document.querySelectorAll("[data-latest-dmg]");
    if (!links.length) return;

    links.forEach(function (link) {
      if (link.dataset.downloadListener === "true") return;
      link.dataset.downloadListener = "true";
      link.addEventListener("click", function (event) {
        if (link.dataset.downloadReady === "true") return;
        event.preventDefault();
        getLatestDmg().then(function (release) {
          window.location.assign(release.url);
        }).catch(function () {
          window.location.assign(releasesUrl);
        });
      });
    });

    getLatestDmg().then(function (release) {
      links.forEach(function (link) {
        link.href = release.url;
        link.dataset.downloadReady = "true";
        link.title = "Download the latest macOS DMG (" + release.version + ")";
        link.setAttribute("aria-label", "Download MLX LoRA Studio " + release.version + " for Mac");
      });
    }).catch(function () {
      links.forEach(function (link) {
        link.href = releasesUrl;
        link.title = "Browse MLX LoRA Studio releases";
      });
    });
  }

  function enhance() {
    configureLatestDownloads();

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
