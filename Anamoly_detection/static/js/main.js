/**
 * UI behavior for the anomaly detection web app.
 * Input mode is driven by a data attribute set from the server (file vs form).
 */
(function () {
  function showInputMethod(method) {
    document.getElementById("file-selector").classList.remove("active");
    document.getElementById("form-selector").classList.remove("active");
    document.getElementById(method + "-selector").classList.add("active");

    document.getElementById("file-input-section").style.display =
      method === "file" ? "block" : "none";
    document.getElementById("form-input-section").style.display =
      method === "form" ? "block" : "none";
  }

  function clearAll() {
    document.getElementById("file-form").reset();
    document.getElementById("manual-form").reset();
    document.getElementById("file-name").textContent = "No file chosen";

    const resultsSection = document.getElementById("results-section");
    if (resultsSection) {
      resultsSection.style.display = "none";
    }
  }

  window.showInputMethod = showInputMethod;
  window.clearAll = clearAll;

  window.addEventListener("DOMContentLoaded", function () {
    var inputType =
      (document.body && document.body.getAttribute("data-input-type")) || "file";
    if (inputType === "form") {
      showInputMethod("form");
    } else {
      showInputMethod("file");
    }

    var fileInput = document.getElementById("file-upload-input");
    var fileNameDisplay = document.getElementById("file-name");
    if (fileInput && fileNameDisplay) {
      fileInput.addEventListener("change", function () {
        if (this.files && this.files.length > 0) {
          fileNameDisplay.textContent = this.files[0].name;
        } else {
          fileNameDisplay.textContent = "No file chosen";
        }
      });
    }
  });
})();
