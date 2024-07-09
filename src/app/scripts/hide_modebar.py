"""Define the JavaScript code to hide the Plotly modebar when it is not needed."""

HIDE_MODEBAR = """document.addEventListener('DOMContentLoaded', function() {
    const scatterplotContainer = document.querySelector('.scatterplot-container');
    const modebar = document.querySelector('.modebar');

    if (scatterplotContainer && modebar) {
        const observer = new MutationObserver(function() {
            if (scatterplotContainer.compareDocumentPosition(modebar) & Node.DOCUMENT_POSITION_FOLLOWING) {
                modebar.style.display = 'none';
            } else {
                modebar.style.display = '';
            }
        });

        observer.observe(scatterplotContainer, { childList: true, subtree: true });
    }
});
"""

__all__ = ["HIDE_MODEBAR"]
