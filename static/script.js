document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('text-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loading = document.getElementById('loading');
    const errorDiv = document.getElementById('error');

    analyzeBtn.addEventListener('click', function() {
        const text = textInput.value.trim();
        
        if (!text) {
            showError('Please enter some text to analyze.');
            return;
        }

        // Hide previous errors
        errorDiv.classList.add('hidden');
        
        // Show loading
        loading.classList.remove('hidden');
        analyzeBtn.disabled = true;

        // Send request to backend
        fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        })
        .then(response => response.json())
        .then(data => {
            loading.classList.add('hidden');
            analyzeBtn.disabled = false;

            if (data.error) {
                showError(data.error);
                return;
            }

            if (data.success) {
                // Redirect to results page
                window.location.href = '/results';
            }
        })
        .catch(error => {
            loading.classList.add('hidden');
            analyzeBtn.disabled = false;
            showError('An error occurred while analyzing the text. Please try again.');
            console.error('Error:', error);
        });
    });

    // Allow Enter key to trigger analysis (Ctrl+Enter or Cmd+Enter)
    textInput.addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.classList.remove('hidden');
    }

    function formatExplanation(text) {
        // Convert markdown-style bold to HTML
        return text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    }
});

// ===============================
// Results Page Logic
// ===============================
document.addEventListener("DOMContentLoaded", function () {
    // ---------- Format explanation text ----------
    const explanationDiv = document.getElementById("explanation-text");
    if (explanationDiv) {
        explanationDiv.innerHTML = explanationDiv.textContent.replace(
            /\*\*(.*?)\*\*/g,
            "<strong>$1</strong>"
        );
    }

    // ---------- Feedback form handling ----------
    const feedbackForm = document.getElementById("feedback-form");

    if (feedbackForm) {
        feedbackForm.addEventListener("submit", async function (e) {
            e.preventDefault();

            const selected = document.querySelector(
                'input[name="feedback_label"]:checked'
            );

            if (!selected) {
                return;
            }

            const feedbackLabel = selected.value;
            const feedbackText = document.getElementById("feedback_text").value;

            try {
                const response = await fetch("/feedback", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        feedback_label: feedbackLabel,
                        feedback_text: feedbackText,
                    }),
                });

                const result = await response.json();
                const status = document.getElementById("feedback-status");

                if (result.success) {
                    status.textContent =
                        "Thank you! Your feedback has been recorded.";
                    status.style.color = "green";
                    feedbackForm.reset();
                } else {
                    status.textContent =
                        "Error submitting feedback. Please try again.";
                    status.style.color = "red";
                }
            } catch (err) {
                console.error("Feedback error:", err);
            }
        });
    }
});
