document.addEventListener('DOMContentLoaded', function() {
    // DOM is fully loaded
    var uploadSpinner = document.getElementById('uploadSpinner');
    var uploadSpinnerText = document.getElementById('uploadSpinnerText');


    // Hide spinner and text on page load
    if (uploadSpinner && uploadSpinnerText) {
        uploadSpinner.style.display = "none";
        uploadSpinnerText.style.display = "none";
    }
});